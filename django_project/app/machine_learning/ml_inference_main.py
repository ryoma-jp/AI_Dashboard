#! -*- coding: utf-8 -*-
"""Standalone inference runner

Runs inference in a separate process to ensure TensorFlow memory is released
when the process exits. Reads config.json, loads the AI Model SDK, performs
predictions for train/validation/test, writes predictions and evaluations
under the model directory.
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


@dataclass
class CocoGroundTruth:
    images: Dict[int, dict]
    annotations: List[dict]
    categories: List[dict]
    filename_to_image: Dict[str, dict]
    gt_lookup: Dict[int, List[dict]]  # image_id -> list of {bbox, category_id}


def _build_coco_gt(dataset_obj, split_name: str) -> CocoGroundTruth:
    """Build COCO-style GT from the DataLoaderCOCO2017 dataframes."""

    if split_name == "train":
        df = getattr(dataset_obj, "df_instances_train", None)
    elif split_name == "validation":
        df = getattr(dataset_obj, "df_instances_validation", None)
    else:
        df = getattr(dataset_obj, "df_instances_test", None)

    if df is None or len(df) == 0:
        return CocoGroundTruth(images={}, annotations=[], categories=[], filename_to_image={}, gt_lookup={})

    images: Dict[int, dict] = {}
    annotations: List[dict] = []
    filename_to_image: Dict[str, dict] = {}
    gt_lookup: Dict[int, List[dict]] = {}
    cat_id_to_name: Dict[int, str] = {}
    ann_id = 1

    for _, row in df.iterrows():
        try:
            image_id = int(row["image_id"])
            width = int(row["width"])
            height = int(row["height"])
        except Exception:
            # Skip malformed rows
            continue

        file_name = row["file_name"]
        bbox = row["bbox"]
        try:
            category_id = int(row["category_id"])
        except Exception:
            continue
        category_name = row["category_name"]

        images.setdefault(image_id, {"id": image_id, "file_name": file_name, "width": width, "height": height})

        # Normalize keys to cope with TFRecord filenames that may include "images/" prefix.
        normalized_keys = {
            str(file_name),
            Path(file_name).name,
            str(Path("images") / Path(file_name).name),
        }
        for k in normalized_keys:
            filename_to_image[k] = images[image_id]
        cat_id_to_name[category_id] = category_name

        if bbox is None or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(max(w, 0.0) * max(h, 0.0)),
                "iscrowd": 0,
            }
        )
        gt_lookup.setdefault(image_id, []).append({"bbox": [float(x), float(y), float(w), float(h)], "category_id": category_id})
        ann_id += 1

    categories = [
        {"id": cid, "name": name}
        for cid, name in sorted(cat_id_to_name.items(), key=lambda x: x[0])
    ]

    return CocoGroundTruth(images=images, annotations=annotations, categories=categories, filename_to_image=filename_to_image, gt_lookup=gt_lookup)


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in xywh format."""

    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _match_tp_fp_fn(gt_boxes: List[dict], pred_boxes: List[dict], iou_thr: float) -> Tuple[int, int, int]:
    """Greedy match predictions to GT to count TP/FP/FN."""

    if not gt_boxes and not pred_boxes:
        return 0, 0, 0

    matched_gt = set()
    tp = 0
    fp = 0

    # Sort predictions by score desc for deterministic matching
    pred_sorted = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)

    for pred in pred_sorted:
        best_iou = 0.0
        best_gt_idx = None
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            if gt["category_id"] != pred["category_id"]:
                continue
            iou = _compute_iou(np.array(gt["bbox"], dtype=float), np.array(pred["bbox"], dtype=float))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx is not None and best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def _boxes_to_xywh(boxes_yxyx: np.ndarray, img_w: int, img_h: int, model_input_size: int) -> np.ndarray:
    """Convert YOLO decoded boxes [y1, x1, y2, x2] to xywh in original image scale."""

    if boxes_yxyx is None or len(boxes_yxyx) == 0:
        return np.zeros((0, 4), dtype=float)

    boxes = np.asarray(boxes_yxyx, dtype=float)
    # Heuristic: if max coordinate <= 1.5, treat as normalized; else treat as model-input pixels.
    if np.max(boxes) <= 1.5:
        x1 = boxes[:, 1] * img_w
        y1 = boxes[:, 0] * img_h
        x2 = boxes[:, 3] * img_w
        y2 = boxes[:, 2] * img_h
    else:
        scale_x = img_w / float(model_input_size)
        scale_y = img_h / float(model_input_size)
        x1 = boxes[:, 1] * scale_x
        y1 = boxes[:, 0] * scale_y
        x2 = boxes[:, 3] * scale_x
        y2 = boxes[:, 2] * scale_y

    x1 = np.clip(x1, 0.0, img_w)
    y1 = np.clip(y1, 0.0, img_h)
    x2 = np.clip(x2, x1, img_w)
    y2 = np.clip(y2, y1, img_h)

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    return np.stack([x1, y1, w, h], axis=1)


def _save_detection_summary(split_name: str, rows: List[dict], evaluation_dir: Path) -> None:
    summary_path = evaluation_dir / f"{split_name}_detection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4, cls=JsonEncoder)


def _save_detection_predictions(split_name: str, predictions: List[dict], evaluation_dir: Path) -> None:
    pred_path = evaluation_dir / f"{split_name}_detection_predictions.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4, cls=JsonEncoder)


def _save_classification_predictions(split_name: str, rows: List[dict], evaluation_dir: Path) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    json_path = evaluation_dir / f"{split_name}_prediction.json"
    csv_path = evaluation_dir / f"{split_name}_prediction.csv"

    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4, cls=JsonEncoder)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def run_split_classification(ai_model_sdk, dataset_iter, split_name, evaluation_dir):
    predictions = []
    targets = []
    filenames = []
    logging.info("start inference: %s", split_name)

    for batch_id, batch in enumerate(dataset_iter):
        if len(batch) == 3:
            inputs, batch_targets, batch_filenames = batch
        else:
            inputs, batch_targets = batch
            batch_filenames = None

        inputs = inputs.numpy()
        batch_targets = batch_targets.numpy()

        pred_raw = ai_model_sdk.predict(inputs, preprocessing=True)
        ai_model_sdk.decode_prediction(pred_raw)

        batch_predictions = [int(cls) for cls in ai_model_sdk.decoded_preds["detection_classes"]]
        batch_targets_list = [int(t) for t in batch_targets.reshape(-1)]

        predictions.extend(batch_predictions)
        targets.extend(batch_targets_list)

        if batch_filenames is not None:
            filenames.extend([fname.decode("utf-8") for fname in batch_filenames.numpy()])

        logging.debug("%s batch %d: inputs=%s preds=%s", split_name, batch_id, inputs.shape, ai_model_sdk.decoded_preds["detection_classes"])

    filenames_out = filenames if filenames else None
    rows = [
        {"id": idx, "prediction": int(p), "target": int(t)}
        for idx, (p, t) in enumerate(zip(predictions, targets))
    ]
    if filenames_out:
        for idx, row in enumerate(rows):
            row["filename"] = filenames_out[idx]

    _save_classification_predictions(split_name, rows, evaluation_dir)

    scores = ai_model_sdk.eval_model(np.array(predictions), np.array(targets))
    logging.info("%s scores: %s", split_name, scores)
    return {f"{split_name} {k}": v for k, v in scores.items()}


def run_split_detection(
    ai_model_sdk,
    dataset_iter,
    split_name: str,
    evaluation_dir: Path,
    dataset_obj,
    dataset_root: Path,
    class_names: List[str],
    model_input_size: int,
    score_thr: float = 0.5,
    iou_thr: float = 0.5,
):
    logging.info("start detection inference: %s", split_name)

    coco_gt = _build_coco_gt(dataset_obj, split_name)
    if not coco_gt.images:
        logging.warning("No ground truth found for split %s", split_name)
        return {f"{split_name} mAP": 0.0, f"{split_name} mAP@0.5": 0.0}, []

    name_to_cat_id = {cat["name"]: cat["id"] for cat in coco_gt.categories}
    predictions: List[dict] = []
    summary_rows: List[dict] = []

    overlay_dir = evaluation_dir / "overlays" / split_name
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for batch_id, batch in enumerate(dataset_iter):
        if len(batch) == 3:
            inputs, _batch_targets, batch_filenames = batch
        else:
            inputs, _batch_targets = batch
            batch_filenames = None

        inputs_np = inputs.numpy()

        pred_raw = ai_model_sdk.predict(inputs_np, preprocessing=True)
        ai_model_sdk.decode_prediction(pred_raw)

        boxes_raw = ai_model_sdk.decoded_preds.get("detection_boxes", [])
        classes_raw = ai_model_sdk.decoded_preds.get("detection_classes", [])
        scores_raw = ai_model_sdk.decoded_preds.get("detection_scores", [])

        filename = None
        if batch_filenames is not None:
            fn_list = [fname.decode("utf-8") for fname in batch_filenames.numpy()]
            filename = fn_list[0] if fn_list else None

        if filename is None:
            logging.warning("Missing filename for batch %s; skipping", batch_id)
            continue

        image_info = coco_gt.filename_to_image.get(filename)
        if image_info is None:
            logging.warning("Filename %s not found in GT; skipping", filename)
            continue

        img_w, img_h = int(image_info["width"]), int(image_info["height"])
        boxes_xywh = _boxes_to_xywh(boxes_raw, img_w, img_h, model_input_size)

        preds_for_image: List[dict] = []
        for box_xywh, cls_idx, score in zip(boxes_xywh, classes_raw, scores_raw):
            score_f = float(score)
            if score_f < score_thr:
                continue
            cls_idx_int = int(cls_idx)
            class_name = class_names[cls_idx_int] if 0 <= cls_idx_int < len(class_names) else str(cls_idx_int)
            category_id = name_to_cat_id.get(class_name, cls_idx_int)
            preds_for_image.append(
                {
                    "image_id": int(image_info["id"]),
                    "category_id": int(category_id),
                    "bbox": [float(box_xywh[0]), float(box_xywh[1]), float(box_xywh[2]), float(box_xywh[3])],
                    "score": score_f,
                }
            )

        predictions.extend(preds_for_image)

        # TP/FP/FN per image
        gt_boxes = coco_gt.gt_lookup.get(int(image_info["id"]), [])
        tp, fp, fn = _match_tp_fp_fn(gt_boxes, preds_for_image, iou_thr)

        # Overlay image
        img_path = dataset_root / split_name / filename
        overlay_rel = Path("evaluations", "overlays", split_name, filename)
        overlay_path = evaluation_dir / "overlays" / split_name / filename
        try:
            image_np = cv2.imread(str(img_path))
            if image_np is not None:
                for pred in preds_for_image:
                    x, y, w, h = pred["bbox"]
                    pt1 = (int(x), int(y))
                    pt2 = (int(x + w), int(y + h))
                    cv2.rectangle(image_np, pt1, pt2, (0, 255, 0), 2)
                    cv2.putText(
                        image_np,
                        f"{pred['category_id']}: {pred['score']:.2f}",
                        (pt1[0], max(0, pt1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(overlay_path), image_np)
        except Exception:
            logging.exception("Failed to create overlay for %s", filename)

        summary_rows.append(
            {
                "data_id": int(image_info["id"]),
                "filename": filename,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "overlay_relpath": str(overlay_rel),
            }
        )

    _save_detection_predictions(split_name, predictions, evaluation_dir)
    _save_detection_summary(split_name, summary_rows, evaluation_dir)

    metrics = {f"{split_name} mAP": 0.0, f"{split_name} mAP@0.5": 0.0}
    try:
        coco = COCO()
        coco.dataset = {
            "images": list(coco_gt.images.values()),
            "annotations": coco_gt.annotations,
            "categories": coco_gt.categories,
        }
        coco.createIndex()

        if predictions:
            coco_dt = coco.loadRes(predictions)
            coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metrics[f"{split_name} mAP"] = float(coco_eval.stats[0])
            metrics[f"{split_name} mAP@0.5"] = float(coco_eval.stats[1])
        else:
            logging.warning("No predictions for split %s; mAP is set to 0", split_name)
    except Exception:
        logging.exception("Failed to compute COCO mAP for split %s", split_name)

    logging.info("%s metrics: %s", split_name, metrics)
    return metrics, summary_rows


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
        return_filename=True,
    )
    val_ds = load_dataset_from_tfrecord(
        task_name,
        dataset_obj.validation_dataset["tfrecord_path"],
        dataset_obj.validation_dataset["class_name_file_path"],
        dataset_obj.validation_dataset["model_input_size"],
        return_filename=True,
    )
    test_ds = load_dataset_from_tfrecord(
        task_name,
        dataset_obj.test_dataset["tfrecord_path"],
        dataset_obj.test_dataset["class_name_file_path"],
        dataset_obj.test_dataset["model_input_size"],
        return_filename=True,
    )

    # Use AI Model SDK batching config when available. For detection, force batch_size=1 to align with decode_prediction logic.
    if task_name == "detection":
        batch_size = 1
        drop_remainder = False
    else:
        batch_size = getattr(dataset_obj, "batch_size", None) or 512
        drop_remainder = True

    train_ds = train_ds.batch(batch_size, drop_remainder=drop_remainder)
    val_ds = val_ds.batch(batch_size, drop_remainder=drop_remainder)
    test_ds = test_ds.batch(batch_size, drop_remainder=drop_remainder)

    model_params = {"model_path": str(model_dir)}
    ai_model_sdk = AI_Model_SDK(str(dataset_pickle), model_params)
    ai_model_sdk.load_model(model_dir / "models")

    evaluation_dir = model_dir / "evaluations"
    evaluations = {}
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Load class names for detection mapping
    with open(dataset_obj.train_dataset["class_name_file_path"], "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    try:
        if task_name == "detection":
            dataset_root = Path(config_data["dataset"]["dataset_dir"]["value"])
            for ds, split in [(train_ds, "train"), (val_ds, "validation"), (test_ds, "test")]:
                metrics, _summary = run_split_detection(
                    ai_model_sdk,
                    ds,
                    split,
                    evaluation_dir,
                    dataset_obj,
                    dataset_root,
                    class_names,
                    dataset_obj.train_dataset["model_input_size"],
                )
                evaluations.update(metrics)
        else:
            evaluations.update(run_split_classification(ai_model_sdk, train_ds, "train", evaluation_dir))
            evaluations.update(run_split_classification(ai_model_sdk, val_ds, "validation", evaluation_dir))
            evaluations.update(run_split_classification(ai_model_sdk, test_ds, "test", evaluation_dir))
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
