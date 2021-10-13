#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import io
import os
import argparse
import pandas as pd

from data_loader.data_loader import DataLoaderMNIST
from data_loader.data_loader import DataLoaderCIFAR10

from trainer.trainer import TrainerMLP, TrainerCNN, TrainerResNet

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='TensorFlowの学習実装サンプル\n'
													'  * No.01: MNISTデータセットを用いた全結合NN学習サンプル\n'
													'  * No.02: CIFAR-10データセットを用いたCNN学習サンプル\n'
													'  * No.03: CIFAR-10データセットを用いたResNet学習サンプル',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--data_type', dest='data_type', type=str, default='CIFAR-10', required=False, \
			help='データ種別(MNIST, CIFAR-10)')
	parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default=None, required=True, \
			help='データセットディレクトリ')
	parser.add_argument('--model_type', dest='model_type', type=str, default='ResNet', required=False, \
			help='モデル種別(MLP, SimpleCNN, DeepCNN, SimpleResNet, DeepResNet)')
	parser.add_argument('--data_augmentation', dest='data_augmentation', type=str, default=None, required=False, \
			help='Data Augmentationパラメータをカンマ区切りで指定\n'
					'  rotation_range,width_shift_range,height_shift_range,horizontal_flip\n'
					'    rotation_range: 回転範囲をdeg単位で指定\n'
					'    width_shift_range: 水平方向のシフト範囲を画像横幅に対する割合で指定\n'
					'    height_shift_range: 垂直方向のシフト範囲を画像縦幅に対する割合で指定\n'
					'    horizontal_flip: 水平方向の反転有無(True or False)')
	parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam', required=False, \
			help='Optimizer(adam(default), sgd, adam_lrs, sgd, lrs)\n'
					'  * lrs: Learning Rate Scheduler')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, required=False, \
			help='ミニバッチサイズ')
	parser.add_argument('--initializer', dest='initializer', type=str, default="glorot_uniform", required=False, \
			help='重みの初期化\n'
					'  glrot_uniform: Xavierの一様分布\n'
					'  glrot_normal: Xavierの正規分布\n'
					'  he_uniform: Heの一様分布\n'
					'  he_normal: Heの正規分布')
	parser.add_argument('--data_norm', dest='data_norm', type=str, default="max", required=False, \
			help='データの正規化手法\n'
					'  max: データの絶対値の最大が1.0となるように正規化(最大値で除算)\n'
					'  max-min: データの最大が1.0最小が0.0となるように正規化(最大値と最小値を用いて算出)\n'
					'  z-score: 標準化(平均と標準偏差を用いて算出)\n')
	parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.0, required=False, \
			help='Dropoutで欠落させるデータの割合')
	parser.add_argument('--loss_func', dest='loss_func', type=str, default='sparse_categorical_crossentropy', required=False, \
			help='コスト関数(tf.keras.lossesのメンバを指定)')
	parser.add_argument('--epochs', dest='epochs', type=int, default=200, required=False, \
			help='学習EPOCH数')
	parser.add_argument('--result_dir', dest='result_dir', type=str, default='./result', required=False, \
			help='学習結果の出力先ディレクトリ')

	args = parser.parse_args()

	return args

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
	print('  * args.data_type = {}'.format(args.data_type))
	print('  * args.dataset_dir = {}'.format(args.dataset_dir))
	print('  * args.model_type = {}'.format(args.model_type))
	print('  * args.data_augmentation = {}'.format(args.data_augmentation))
	print('  * args.optimizer = {}'.format(args.optimizer))
	print('  * args.batch_size = {}'.format(args.batch_size))
	print('  * args.initializer = {}'.format(args.initializer))
	print('  * args.data_norm = {}'.format(args.data_norm))
	print('  * args.dropout_rate = {}'.format(args.dropout_rate))
	print('  * args.loss_func = {}'.format(args.loss_func))
	print('  * args.epochs = {}'.format(args.epochs))
	print('  * args.result_dir = {}'.format(args.result_dir))
	
	# --- Data Augmentationパラメータを辞書型に変換 ---
	if (args.data_augmentation is not None):
		dict_keys = ['rotation_range', 'width_shift_range', 'height_shift_range', 'zoom_range', 'channel_shift_range', 'horizontal_flip']
		df_da_params = pd.read_csv(io.StringIO(args.data_augmentation), header=None, skipinitialspace=True).values[0]
		
		data_augmentation = {}
		for (key, da_param) in zip(dict_keys, df_da_params):
			data_augmentation[key] = da_param
	else:
		data_augmentaion = None
	
	if (args.loss_func == "sparse_categorical_crossentropy"):
		one_hot = False
	else:
		one_hot = True
	if (args.data_type == "MNIST"):
		dataset = DataLoaderMNIST(args.dataset_dir, validation_split=0.2, one_hot=one_hot)
	elif (args.data_type == "CIFAR-10"):
		dataset = DataLoaderCIFAR10(args.dataset_dir, validation_split=0.2, one_hot=one_hot)
	else:
		print('[ERROR] Unknown data_type: {}'.format(args.data_type))
		quit()
		
	print_ndarray_shape(dataset.train_images)
	print_ndarray_shape(dataset.train_labels)
	print_ndarray_shape(dataset.validation_images)
	print_ndarray_shape(dataset.validation_labels)
	print_ndarray_shape(dataset.test_images)
	print_ndarray_shape(dataset.test_labels)
	
	x_train, x_val, x_test = dataset.normalization(args.data_norm)
	y_train = dataset.train_labels
	y_val = dataset.validation_labels
	y_test = dataset.test_labels
	output_dims = dataset.output_dims
	
	if (args.model_type == 'MLP'):
		trainer = TrainerMLP(dataset.train_images.shape[1:], output_dir=args.result_dir,
			optimizer=args.optimizer, initializer=args.initializer)
	elif (args.model_type == 'SimpleCNN'):
		trainer = TrainerCNN(dataset.train_images.shape[1:], output_dir=args.result_dir,
			optimizer=args.optimizer, loss=args.loss_func, initializer=args.initializer)
	elif (args.model_type == 'DeepCNN'):
		trainer = TrainerCNN(dataset.train_images.shape[1:], output_dir=args.result_dir,
			optimizer=args.optimizer, loss=args.loss_func, initializer=args.initializer, model_type='deep_model')
	elif (args.model_type == 'SimpleResNet'):
		trainer = TrainerResNet(dataset.train_images.shape[1:], output_dims, output_dir=args.result_dir,
			model_type='custom', 
			optimizer=args.optimizer, loss=args.loss_func, initializer=args.initializer, dropout_rate=args.dropout_rate)
	elif (args.model_type == 'DeepResNet'):
		trainer = TrainerResNet(dataset.train_images.shape[1:], output_dims, output_dir=args.result_dir,
			model_type='custom_deep', 
			optimizer=args.optimizer, loss=args.loss_func, initializer=args.initializer, dropout_rate=args.dropout_rate)
	else:
		print('[ERROR] Unknown model_type: {}'.format(args.model_type))
		quit()
	trainer.fit(x_train, y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
		batch_size=args.batch_size, da_params=data_augmentation, epochs=args.epochs)
	trainer.save_model()
	
	predictions = trainer.predict(x_test)
	print('\nPredictions(shape): {}'.format(predictions.shape))

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

