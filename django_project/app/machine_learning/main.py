#! -*- coding: utf-8 -*-
'''DeepLearning学習処理の実装サンプル

引数に指定する設定ファイルで指定されたパラメータに従い，DeepLearningモデルの学習を実行する実装サンプル．

設定ファイルで指定するパラメータ:

  * env: 環境設定
    * fifo: 学習制御用のFIFOパス
    * result_dir: 結果を格納するディレクトリ
  * dataset: データセット関連の設定
    * dataset_name: データセット名(Preset: MNIST, CIFAR-10)
    * dataset_dir: データセットを格納したディレクトリ
    * norm: 正規化方式(max, max-min, z-score)
    * data_augmentation: DataAugmentation関連の設定
      * rotation_range: 画像の回転[deg]
      * width_shift_range: 水平方向の画像幅に対するシフト率[0.0-1.0]
      * height_shift_range: 垂直方向の画像高さに対するシフト率[0.0-1.0]
      * zoom_range: 拡大率[%]
      * channel_shift_range: チャネル(RGB)のシフト率[0.0-1.0]
      * horizontal_flip: 水平方向反転有無(True or False)
  * model: 学習するモデル関連の設定
    * model_type: モデル種別(MLP, SimpleCNN, DeepCNN, SimpleResNet, DeepResNet)
  * training_parameter: ハイパーパラメータ
    * optimizer: 最適化方式(momentum, adam, sgd, adam_lrs, sgd, lrs)
    * batch_size: バッチサイズ
    * epochs: EPOCH数
    * initializer: 重みの初期化アルゴリズム
        glrot_uniform: Xavierの一様分布
        glrot_normal: Xavierの正規分布
        he_uniform: Heの一様分布
        he_normal: Heの正規分布
    * droptout_rate: ドロップアウトによる欠落率[0.0-1.0]
    * loss_func: 損失関数(tf.keras.lossesのメンバを指定)
    
'''

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

from machine_learning.lib.data_loader.data_loader import DataLoaderMNIST
from machine_learning.lib.data_loader.data_loader import DataLoaderCIFAR10

from machine_learning.lib.trainer.trainer import TrainerMLP, TrainerCNN, TrainerResNet

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
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
	predictions = trainer.predict(x)
	print('\nPredictions(shape): {}'.format(predictions.shape))
	
	if (y is not None):
		predictions_idx = np.argmax(predictions, axis=1)
		y_idx = np.argmax(y, axis=1)
		
		print('n_data : {}'.format(len(predictions_idx)))
		print('n_correct : {}'.format(len(predictions_idx[predictions_idx==y_idx])))
		
	return predictions

def main():
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
	data_augmentation = {}
	for (key, value) in config_data['dataset']['data_augmentation'].items():
		data_augmentation[key] = value['value']
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
	
	# --- データセット読み込み ---
	with open(Path(dataset_dir, 'dataset.pkl'), 'rb') as f:
		dataset = pickle.load(f)
	
	if (loss_func == "sparse_categorical_crossentropy"):
		one_hot = False
	else:
		one_hot = True
	dataset.convert_label_encoding(one_hot=one_hot)
	
	print_ndarray_shape(dataset.train_images)
	print_ndarray_shape(dataset.train_labels)
	print_ndarray_shape(dataset.validation_images)
	print_ndarray_shape(dataset.validation_labels)
	print_ndarray_shape(dataset.test_images)
	print_ndarray_shape(dataset.test_labels)
	
	x_train, x_val, x_test = dataset.normalization(data_norm)
	y_train = dataset.train_labels
	y_val = dataset.validation_labels
	y_test = dataset.test_labels
	output_dims = dataset.output_dims
	
	# --- モデル取得 ---
	if (args.mode == 'predict'):
		model_file = Path(result_dir, 'models', 'hdf5', 'model.h5')
		if (not model_file.exists()):
			model_file = None
	else:
		model_file = None
	
	if (model_type == 'MLP'):
		trainer = TrainerMLP(dataset.train_images.shape[1:],
			output_dir=result_dir, model_file=model_file,
			optimizer=optimizer, initializer=initializer)
	elif (model_type == 'SimpleCNN'):
		trainer = TrainerCNN(dataset.train_images.shape[1:],
			output_dir=result_dir, model_file=model_file,
			optimizer=optimizer, loss=loss_func, initializer=initializer)
	elif (model_type == 'DeepCNN'):
		trainer = TrainerCNN(dataset.train_images.shape[1:],
			output_dir=result_dir, model_file=model_file,
			optimizer=optimizer, loss=loss_func, initializer=initializer, model_type='deep_model')
	elif (model_type == 'SimpleResNet'):
		trainer = TrainerResNet(dataset.train_images.shape[1:], output_dims,
			output_dir=result_dir, model_file=model_file,
			model_type='custom', 
			optimizer=optimizer, loss=loss_func, initializer=initializer, dropout_rate=dropout_rate)
	elif (model_type == 'DeepResNet'):
		trainer = TrainerResNet(dataset.train_images.shape[1:], output_dims,
			output_dir=result_dir, model_file=model_file,
			model_type='custom_deep', 
			optimizer=optimizer, loss=loss_func, initializer=initializer, dropout_rate=dropout_rate)
	else:
		print('[ERROR] Unknown model_type: {}'.format(model_type))
		quit()
	
	if (args.mode == 'train'):
		# --- 学習 ---
		trainer.fit(x_train, y_train,
			x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
			web_app_ctrl_fifo=web_app_ctrl_fifo, trainer_ctrl_fifo=trainer_ctrl_fifo, 
			batch_size=batch_size, da_params=data_augmentation, epochs=epochs)
		trainer.save_model()
		
		predictions = _predict_and_calc_accuracy(trainer, x_test, y_test)
	elif (args.mode == 'predict'):
		predictions = _predict_and_calc_accuracy(trainer, x_test, y_test)
		
		json_data = []
		for i, (prediction, label) in enumerate(zip(np.argmax(predictions, axis=1), np.argmax(y_test, axis=1))):
			json_data.append({
				'id': int(i),
				'prediction': int(prediction),
				'label': int(label),
			})
		with open(Path(result_dir, 'prediction.json'), 'w') as f:
			json.dump(json_data, f, ensure_ascii=False, indent=4)
		
	else:
		print('[ERROR] Unknown mode: {}'.format(args.mode))

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

