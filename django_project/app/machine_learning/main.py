#! -*- coding: utf-8 -*-

"""DeepLearning学習処理の実装サンプル

引数に指定する設定ファイルで指定されたパラメータに従い，DeepLearningモデルの学習を実行する実装サンプル．

設定ファイルで指定するパラメータ:

- env: 環境設定

  - fifo: 学習制御用のFIFOパス
  - result_dir: 結果を格納するディレクトリ

- dataset: データセット関連の設定

  - dataset_name: データセット名(Preset: MNIST, CIFAR-10)
  - dataset_dir: データセットを格納したディレクトリ
  - norm: 正規化方式(max, max-min, z-score)
  - image_data_augmentation: DataAugmentation関連の設定
  
    - rotation_range: 画像の回転[deg]
    - width_shift_range: 水平方向の画像幅に対するシフト率[0.0-1.0]
    - height_shift_range: 垂直方向の画像高さに対するシフト率[0.0-1.0]
    - zoom_range: 拡大率[%]
    - channel_shift_range: チャネル(RGB)のシフト率[0.0-1.0]
    - horizontal_flip: 水平方向反転有無(True or False)
    
- model: 学習するモデル関連の設定

  - model_type: モデル種別(MLP, SimpleCNN, DeepCNN, SimpleResNet, DeepResNet)
  
- training_parameter: ハイパーパラメータ

  - optimizer: 最適化方式(momentum, adam, sgd, adam_lrs, sgd, lrs)
  - batch_size: バッチサイズ
  - epochs: EPOCH数
  - initializer: 重みの初期化アルゴリズム
  
      - glrot_uniform: Xavierの一様分布
      - glrot_normal: Xavierの正規分布
      - he_uniform: Heの一様分布
      - he_normal: Heの正規分布
      
  - droptout_rate: ドロップアウトによる欠落率[0.0-1.0]
  - loss_func: 損失関数(tf.keras.lossesのメンバを指定)
  
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
    loss_func = config_data['training_parameter']['loss_func']['value']
    optimizer = config_data['training_parameter']['optimizer']['value']
    initializer = config_data['training_parameter']['initializer']['value']
    dropout_rate = config_data['training_parameter']['dropout_rate']['value']
    batch_size = config_data['training_parameter']['batch_size']['value']
    epochs = config_data['training_parameter']['epochs']['value']
    learning_rate = config_data['training_parameter']['learning_rate']['value']
    
    # --- データセット読み込み ---
    with open(Path(dataset_dir, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
        if (loss_func == "sparse_categorical_crossentropy"):
            one_hot = False
        else:
            one_hot = True
        dataset.convert_label_encoding(one_hot=one_hot)
    
    print_ndarray_shape(dataset.train_x)
    print_ndarray_shape(dataset.train_y)
    print_ndarray_shape(dataset.validation_x)
    print_ndarray_shape(dataset.validation_y)
    print_ndarray_shape(dataset.test_x)
    print_ndarray_shape(dataset.test_y)
    
    x_train, x_val, x_test = dataset.normalization(data_norm)
    y_train = dataset.train_y
    y_val = dataset.validation_y
    y_test = dataset.test_y
    output_dims = dataset.output_dims
    
    # --- モデル取得 ---
    if (args.mode == 'predict'):
        if (model_type in ['MLP', 'SimpleCNN', 'DeepCNN', 'SimpleResNet', 'DeepResNet']):
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
        trainer = TrainerKerasMLP(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file,
            optimizer=optimizer, initializer=initializer,
            learning_rate=learning_rate)
    elif (model_type == 'SimpleCNN'):
        trainer = TrainerKerasCNN(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file,
            optimizer=optimizer, loss=loss_func, initializer=initializer,
            learning_rate=learning_rate)
    elif (model_type == 'DeepCNN'):
        trainer = TrainerKerasCNN(dataset.train_x.shape[1:], classes=output_dims,
            output_dir=result_dir, model_file=model_file,
            optimizer=optimizer, loss=loss_func, initializer=initializer, model_type='deep_model',
            learning_rate=learning_rate)
    elif (model_type == 'SimpleResNet'):
        trainer = TrainerKerasResNet(dataset.train_x.shape[1:], output_dims,
            output_dir=result_dir, model_file=model_file,
            model_type='custom', 
            optimizer=optimizer, loss=loss_func, initializer=initializer, dropout_rate=dropout_rate,
            learning_rate=learning_rate)
    elif (model_type == 'DeepResNet'):
        trainer = TrainerKerasResNet(dataset.train_x.shape[1:], output_dims,
            output_dir=result_dir, model_file=model_file,
            model_type='custom_deep', 
            optimizer=optimizer, loss=loss_func, initializer=initializer, dropout_rate=dropout_rate,
            learning_rate=learning_rate)
    elif (model_type == 'LightGBM'):
        trainer = TrainerLightGBM(output_dir=result_dir, model_file=model_file,
            learning_rate=learning_rate)
    else:
        print('[ERROR] Unknown model_type: {}'.format(model_type))
        quit()
    
    if (args.mode == 'train'):
        # --- 学習 ---
        trainer.fit(x_train, y_train,
            x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
            web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
            batch_size=batch_size, da_params=image_data_augmentation, epochs=epochs)
        trainer.save_model()
        
        if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
            predictions = _predict_and_calc_accuracy(trainer, x_test, y_test)
    
    elif (args.mode == 'predict'):
        predict_data_list = [
            ['train', x_train, y_train],
            ['validation', x_val, y_val],
            ['test', x_test, y_test],
        ]
        
        for (name, x_, y_) in predict_data_list:
            if (x_ is not None):
                if ((dataset.dataset_type == 'img_clf') or (dataset.dataset_type == 'table_clf')):
                    predictions = _predict_and_calc_accuracy(trainer, x_, y_)
                    
                    json_data = []
                    if (y_ is not None):
                        for i, (prediction, target) in enumerate(zip(np.argmax(predictions, axis=1), np.argmax(y_, axis=1))):
                            json_data.append({
                                'id': int(i),
                                'prediction': int(prediction),
                                'target': int(target),
                            })
                    else:
                        for i, prediction in enumerate(np.argmax(predictions, axis=1)):
                            json_data.append({
                                'id': int(i),
                                'prediction': int(prediction),
                                'target': '(no data)',
                            })
                else:
                    predictions = trainer.predict(x_)
                    
                    json_data = []
                    if (y_ is not None):
                        for i, (prediction, target) in enumerate(zip(predictions, y_.values.reshape(-1))):
                            if ('id' in x_.columns):
                                sample_id = x_['id'].iloc[i]
                            else:
                                sample_id = i
                            
                            json_data.append({
                                'id': sample_id,
                                'prediction': prediction,
                                'target': target,
                            })
                        
                    else:
                        for i, prediction in enumerate(predictions):
                            if ('id' in x_.columns):
                                sample_id = x_['id'].iloc[i]
                            else:
                                sample_id = i
                            
                            json_data.append({
                                'id': int(i),
                                'prediction': prediction,
                                'target': "(no data)",
                            })
                
                with open(Path(result_dir, f'{name}_prediction.json'), 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    else:
        print('[ERROR] Unknown mode: {}'.format(args.mode))

    return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
    main()

