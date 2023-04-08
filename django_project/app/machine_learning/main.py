#! -*- coding: utf-8 -*-

"""機械学習学習処理

引数に指定する設定ファイルで指定されたパラメータに従い，モデルの学習を実行する

"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import json
import argparse
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from pathlib import Path

from machine_learning.lib.trainer.trainer_keras import TrainerKerasMLP, TrainerKerasCNN, TrainerKerasResNet
from machine_learning.lib.trainer.trainer_lgb import TrainerLightGBM

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
    """ArgParser
    
    引数を読み込む
    
    """
    
    parser = argparse.ArgumentParser(description='TensorFlowの学習実装サンプル',
                formatter_class=argparse.RawTextHelpFormatter)

    # --- 引数を追加 ---
    parser.add_argument('--mode', dest='mode', type=str, default=None, required=True, \
            help='機械学習の動作モードを選択("train", "predict")')
    parser.add_argument('--config', dest='config', type=str, default=None, required=True, \
            help='設定ファイル(*.json)')

    args = parser.parse_args()

    return args

def _predict_and_calc_accuracy(trainer, x, y=None):
    """_predict_and_calc_accuracy
    
    推論および精度計算を実行する
    
    精度計算は真値(y)が指定された場合のみ実行する
    
    Args:
        trainer (Trainer): 学習済みモデル．Trainerクラスまたは派生クラスインスタンスを指定する．
        x (numpy.ndarray): 推論対象データの入力値
        y (:obj:`numpy.ndarray`, optional): 推論対象データの真値
    
    """
    predictions = trainer.predict(x)
    print('\nPredictions(shape): {}'.format(predictions.shape))
    
    if (y is not None):
        predictions_idx = np.argmax(predictions, axis=1)
        y_idx = np.argmax(y, axis=1)
        
        print('n_data : {}'.format(len(predictions_idx)))
        print('n_correct : {}'.format(len(predictions_idx[predictions_idx==y_idx])))
        
    return predictions

def main():
    """main
    
    メイン関数
    
    """
    # --- NumPy配列形状表示 ---
    def print_ndarray_shape(ndarr):
        if (ndarr is not None):
            print(ndarr.shape)
        else:
            pass
        return
        
    # --- 引数処理 ---
    args = ArgParser()
    print('[INFO] Arguments')
    print('  * args.mode = {}'.format(args.mode))
    print('  * args.config = {}'.format(args.config))
    
    # --- configファイルをロード ---
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    print('[INFO] Config data')
    print(json.dumps(config_data, indent=2))
    
    # --- 固定パラメータ ---
    DNN_MODEL_LIST = ['MLP', 'SimpleCNN', 'DeepCNN', 'SimpleResNet', 'DeepResNet']
    
    # --- 設定パラメータを取得 ---
    web_app_ctrl_fifo = config_data['env']['web_app_ctrl_fifo']['value']
    trainer_ctrl_fifo = config_data['env']['trainer_ctrl_fifo']['value']
    result_dir = config_data['env']['result_dir']['value']
    image_data_augmentation = {}
    for (key, value) in config_data['dataset']['image_data_augmentation'].items():
        image_data_augmentation[key] = value['value']
    data_type = config_data['dataset']['dataset_name']['value']
    dataset_dir = config_data['dataset']['dataset_dir']['value']
    data_norm = config_data['dataset']['norm']['value']
    model_type = config_data['model']['model_type']['value']
    
    if (model_type in DNN_MODEL_LIST):
        loss_func = config_data['dnn_training_parameter']['loss_func']['value']
        if (loss_func == "sparse_categorical_crossentropy"):
            one_hot = False
        else:
            one_hot = True
        
        optimizer = config_data['dnn_training_parameter']['optimizer']['value']
        initializer = config_data['dnn_training_parameter']['initializer']['value']
        dropout_rate = config_data['dnn_training_parameter']['dropout_rate']['value']
        batch_size = config_data['dnn_training_parameter']['batch_size']['value']
        epochs = config_data['dnn_training_parameter']['epochs']['value']
        learning_rate = config_data['dnn_training_parameter']['learning_rate']['value']
    elif (model_type == 'LightGBM'):
        num_leaves = config_data['lgb_training_parameter']['num_leaves']['value']
        max_depth = config_data['lgb_training_parameter']['max_depth']['value']
        learning_rate = config_data['lgb_training_parameter']['learning_rate']['value']
        feature_fraction = config_data['lgb_training_parameter']['feature_fraction']['value']
        bagging_fraction = config_data['lgb_training_parameter']['bagging_fraction']['value']
        bagging_freq = config_data['lgb_training_parameter']['bagging_freq']['value']
        lambda_l1 = config_data['lgb_training_parameter']['lambda_l1']['value']
        lambda_l2 = config_data['lgb_training_parameter']['lambda_l2']['value']
        boosting = config_data['lgb_training_parameter']['boosting']['value']
    
    # --- データセット読み込み ---
    with open(Path(dataset_dir, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
        dataset.convert_label_encoding(one_hot=one_hot)
    
    print_ndarray_shape(dataset.train_x)
    print_ndarray_shape(dataset.train_y)
    print_ndarray_shape(dataset.validation_x)
    print_ndarray_shape(dataset.validation_y)
    print_ndarray_shape(dataset.test_x)
    print_ndarray_shape(dataset.test_y)
    
    x_train, y_train, x_val, y_val, x_test, y_test = dataset.preprocessing(norm_mode=data_norm)
    output_dims = dataset.output_dims
    
    # --- モデル取得 ---
    if (args.mode == 'predict'):
        if (model_type in DNN_MODEL_LIST):
            model_file = Path(result_dir, 'models', 'hdf5', 'model.h5')
            if (not model_file.exists()):
                model_file = None
        elif (model_type in ['LightGBM']):
            model_file = Path(result_dir, 'models', 'lightgbm_model.pickle')
            if (not model_file.exists()):
                model_file = None
        else:
            model_file = None
    else:
        model_file = None
    
    if (model_type == 'MLP'):
        print('Create MLP')
        
        num_of_hidden_nodes = config_data['mlp_structure']['num_of_hidden_nodes']['value']
        
        trainer = TrainerKerasMLP(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file,
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            initializer=initializer, optimizer=optimizer, loss=loss_func,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            dataset_type=dataset.dataset_type, da_params=image_data_augmentation,
            batch_size=batch_size, epochs=epochs,
            num_of_hidden_nodes=num_of_hidden_nodes)
    elif (model_type == 'SimpleCNN'):
        print('Create SimpleCNN')
        trainer = TrainerKerasCNN(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file,
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            initializer=initializer, optimizer=optimizer, loss=loss_func,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            dataset_type=dataset.dataset_type, da_params=image_data_augmentation,
            batch_size=batch_size, epochs=epochs)
    elif (model_type == 'DeepCNN'):
        print('Create DeepCNN')
        trainer = TrainerKerasCNN(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file, model_type='deep_model',
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            initializer=initializer, optimizer=optimizer, loss=loss_func,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            dataset_type=dataset.dataset_type, da_params=image_data_augmentation,
            batch_size=batch_size, epochs=epochs)
    elif (model_type == 'SimpleResNet'):
        print('Create SimpleResNet')
        trainer = TrainerKerasResNet(dataset.train_x.shape[1:], output_dims,
            output_dir=result_dir, model_file=model_file, model_type='custom', 
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            initializer=initializer, optimizer=optimizer, loss=loss_func,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            dataset_type=dataset.dataset_type, da_params=image_data_augmentation,
            batch_size=batch_size, epochs=epochs)
    elif (model_type == 'DeepResNet'):
        print('Create DeepResNet')
        trainer = TrainerKerasResNet(dataset.train_x.shape[1:], output_dims,
            output_dir=result_dir, model_file=model_file, model_type='custom_deep', 
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            initializer=initializer, optimizer=optimizer, loss=loss_func,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            dataset_type=dataset.dataset_type, da_params=image_data_augmentation,
            batch_size=batch_size, epochs=epochs)
    elif (model_type == 'LightGBM'):
        print('Create LightGBM')
        trainer = TrainerLightGBM(output_dir=result_dir, model_file=model_file,
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            num_leaves=num_leaves, max_depth=max_depth,
            learning_rate=learning_rate, feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction, bagging_freq=bagging_freq,
            lambda_l1=lambda_l1, lambda_l2=lambda_l2, boosting=boosting)
    else:
        print('[ERROR] Unknown model_type: {}'.format(model_type))
        quit()
    
    if (args.mode == 'train'):
        # --- model training and save ---
        trainer.fit(x_train, y_train,
                    x_val=x_val, y_val=y_val,
                    x_test=x_test, y_test=y_test)
        trainer.save_model()
        
        # --- update config file ---
        config_data['inference_parameter']['preprocessing']['norm_coef_a']['value'] = dataset.preprocessing_params['norm_coef'][0]
        config_data['inference_parameter']['preprocessing']['norm_coef_b']['value'] = dataset.preprocessing_params['norm_coef'][1]
        config_data['inference_parameter']['preprocessing']['input_shape']['value'] = dataset.train_x.shape[1:]
        config_data['inference_parameter']['model']['task']['value'] = dataset.dataset_type
        with open(args.config, 'w') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)

        # --- save feature importance as json ---
        if (model_type == 'LightGBM'):
            df_importance = trainer.get_importance(index=x_train.columns)
            dict_importance = df_importance.sort_values('importance', ascending=False).to_dict(orient='index')
            with open(Path(result_dir, 'feature_importance.json'), 'w') as f:
                json.dump(dict_importance, f, ensure_ascii=False, indent=4)
        
        if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
            predictions = _predict_and_calc_accuracy(trainer, x_test, y_test)
    
    elif (args.mode == 'predict'):
        predict_data_list = [
            ['train', x_train, y_train],
            ['validation', x_val, y_val],
            ['test', x_test, y_test],
        ]
        
        for (name, x_, y_) in predict_data_list:
            print(f'[INFO] Prediction: {name}')
            if (x_ is not None):
                if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
                    predictions = _predict_and_calc_accuracy(trainer, x_, y_)
                    
                    json_data = []
                    if (y_ is not None):
                        for i, (prediction, target) in enumerate(tqdm(zip(np.argmax(predictions, axis=1), np.argmax(y_, axis=1)))):
                            json_data.append({
                                'id': int(i),
                                'prediction': int(prediction),
                                'target': int(target),
                            })
                    else:
                        for i, prediction in enumerate(tqdm(np.argmax(predictions, axis=1))):
                            json_data.append({
                                'id': int(i),
                                'prediction': int(prediction),
                                'target': '(no data)',
                            })
                else:
                    predictions = trainer.predict(x_)
                    
                    json_data = []
                    if (y_ is not None):
                        for i, (prediction, target) in enumerate(tqdm(zip(predictions, y_.values.reshape(-1)))):
                            if ('id' in x_.columns):
                                sample_id = x_['id'].iloc[i]
                            else:
                                sample_id = i
                            
                            json_data.append({
                                'id': sample_id,
                                'prediction': float(prediction),
                                'target': target,
                            })
                        
                    else:
                        for i, prediction in enumerate(tqdm(predictions)):
                            if ('id' in x_.columns):
                                sample_id = x_['id'].iloc[i]
                            else:
                                sample_id = i
                            
                            json_data.append({
                                'id': int(i),
                                'prediction': float(prediction),
                                'target': "(no data)",
                            })
                
                with open(Path(result_dir, f'{name}_prediction.json'), 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                pd.DataFrame(json_data).to_csv(Path(result_dir, f'{name}_prediction.csv'), index=False)
        
    else:
        print('[ERROR] Unknown mode: {}'.format(args.mode))

    return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
    main()

