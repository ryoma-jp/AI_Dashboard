#! -*- coding: utf-8 -*-
"""Standalone inference runner

Runs inference in a separate process to ensure TensorFlow memory is released
when the process exits. Reads config.json, loads the AI Model SDK, performs
predictions for train/validation/test, writes predictions and evaluations
under the model directory.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from machine_learning.lib.data_loader.data_loader import load_dataset_from_tfrecord
from machine_learning.lib.utils.utils import JsonEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with AI Model SDK in a separate process",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--sdk_path", required=True, help="Path to AI Model SDK directory")
    parser.add_argument("--config", required=True, help="Path to config.json")
    return parser.parse_args()


def save_predictions(split_name, predictions, targets, evaluation_dir):
    json_data = []
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        json_data.append({"id": idx, "prediction": int(pred), "target": int(target)})

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    json_path = evaluation_dir / f"{split_name}_prediction.json"
    csv_path = evaluation_dir / f"{split_name}_prediction.csv"

    with open(json_path, "w") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
    pd.DataFrame(json_data).to_csv(csv_path, index=False)


def run_split(ai_model_sdk, dataset_iter, split_name, evaluation_dir):
    predictions = []
    targets = []
    logging.info("start inference: %s", split_name)

    for batch_id, batch in enumerate(dataset_iter):
        inputs = batch[0].numpy()
        batch_targets = batch[1].numpy()

        pred_raw = ai_model_sdk.predict(inputs, preprocessing=True)
        ai_model_sdk.decode_prediction(pred_raw)

        predictions.extend([int(cls) for cls in ai_model_sdk.decoded_preds["detection_classes"]])
        targets.extend([int(t) for t in batch_targets.reshape(-1)])

        logging.debug("%s batch %d: inputs=%s preds=%s", split_name, batch_id, inputs.shape, ai_model_sdk.decoded_preds["detection_classes"])

    save_predictions(split_name, predictions, targets, evaluation_dir)

    scores = ai_model_sdk.eval_model(np.array(predictions), np.array(targets))
    logging.info("%s scores: %s", split_name, scores)
    return {f"{split_name} {k}": v for k, v in scores.items()}


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config_data = json.load(f)

    model_dir = Path(config_data["env"]["result_dir"]["value"]).resolve()
    dataset_pickle = Path(config_data["dataset"]["dataset_dir"]["value"], "dataset.pkl").resolve()

    sys.path.append(args.sdk_path)
    try:
        from ai_model_sdk import AI_Model_SDK
    except Exception:
        logging.exception("failed to import AI_Model_SDK")
        raise

    task_table = {"img_clf": "classification", "img_det": "detection"}

    with open(dataset_pickle, "rb") as f:
        dataset_obj = pickle.load(f)
    task_key = config_data["inference_parameter"]["model"]["task"]["value"]
    task_name = task_table.get(task_key, "classification")

    train_ds = load_dataset_from_tfrecord(
        task_name,
        dataset_obj.train_dataset["tfrecord_path"],
        dataset_obj.train_dataset["class_name_file_path"],
        dataset_obj.train_dataset["model_input_size"],
    )
    val_ds = load_dataset_from_tfrecord(
        task_name,
        dataset_obj.validation_dataset["tfrecord_path"],
        dataset_obj.validation_dataset["class_name_file_path"],
        dataset_obj.validation_dataset["model_input_size"],
    )
    test_ds = load_dataset_from_tfrecord(
        task_name,
        dataset_obj.test_dataset["tfrecord_path"],
        dataset_obj.test_dataset["class_name_file_path"],
        dataset_obj.test_dataset["model_input_size"],
    )

    # Use AI Model SDK batching config when available
    batch_size = getattr(dataset_obj, "batch_size", None) or 512
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)

    model_params = {"model_path": str(model_dir)}
    ai_model_sdk = AI_Model_SDK(str(dataset_pickle), model_params)
    ai_model_sdk.load_model(model_dir / "models")

    evaluation_dir = model_dir / "evaluations"
    evaluations = {}
    try:
        evaluations.update(run_split(ai_model_sdk, train_ds, "train", evaluation_dir))
        evaluations.update(run_split(ai_model_sdk, val_ds, "validation", evaluation_dir))
        evaluations.update(run_split(ai_model_sdk, test_ds, "test", evaluation_dir))
    finally:
        with open(evaluation_dir / "evaluations.json", "w") as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=4, cls=JsonEncoder)

        # Best-effort cleanup before process exit
        try:
            tf.keras.backend.clear_session()
        except Exception:
            logging.exception("failed to clear TF session")
        gc.collect()

        # Remove SDK import path to reduce sys.path pollution in reuse scenarios
        if args.sdk_path in sys.path:
            sys.path.remove(args.sdk_path)


if __name__ == "__main__":
    main()
