#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import fcntl
import gc
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#---------------------------------
# 環境変数設定
#---------------------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

#---------------------------------
# クラス; 学習モジュール基底クラス
#---------------------------------
class Trainer():
	# --- カスタムコールバック ---
	class CustomCallback(keras.callbacks.Callback):
		def __init__(self, trainer_ctrl_fifo):
			super().__init__()
			self.trainer_ctrl_fifo = trainer_ctrl_fifo
			
		def on_train_batch_end(self, batch, logs=None):
			if (self.trainer_ctrl_fifo is not None):
				fd = os.open(self.trainer_ctrl_fifo, os.O_RDONLY | os.O_NONBLOCK)
				flags = fcntl.fcntl(fd, fcntl.F_GETFL)
				flags &= ~os.O_NONBLOCK
				fcntl.fcntl(fd, fcntl.F_SETFL, flags)
				
				try:
					command = os.read(fd, 128)
					command = command.decode()[:-1]
					while (True):
						buf = os.read(fd, 65536)
						if not buf:
							break
				finally:
					os.close(fd)
			
				if (command):
					if (command == 'stop'):
						print('End batch: recv command={}'.format(command))
						self.model.stop_training = True
					else:
						print('End batch: recv unknown command={}'.format(command))
			
		def on_epoch_end(self, epoch, logs=None):
			keys = list(logs.keys())
			log_str = ''
			for key in keys[:-1]:
				log_str += '({} = {}), '.format(key, logs[key])
			log_str += '({} = {})'.format(keys[-1], logs[keys[-1]])
			print("End epoch {}: {}".format(epoch, log_str))
			
	# --- コンストラクタ ---
	def __init__(self, output_dir=None, model_file=None, optimizer='adam', loss='sparse_categorical_crossentropy'):
		# --- 出力ディレクトリ作成 ---
		self.output_dir = output_dir
		if (output_dir is not None):
			os.makedirs(output_dir, exist_ok=True)
		
		# --- モデル構築 ---
		def _load_model(model_file):
			if (model_file is not None):
				return keras.models.load_model(model_file)
			else:
				return None
		
		self.model = _load_model(model_file)
		if (self.model is not None):
			self._compile_model(optimizer=optimizer, loss=loss)
		
		return
	
	# --- モデルの構成 ---
	#   * lr_decay: 学習率減衰する(=True)，しない(=False; default)を指定
	def _compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy'):
		
		if (optimizer == 'adam'):
			opt = tf.keras.optimizers.Adam()
		elif (optimizer == 'sgd'):
			opt = tf.keras.optimizers.SGD()
		elif (optimizer == 'adam_lrs'):
			# --- parameters ---
			#  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
			#    but, initial learning rate is default of Adam()
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
				0.001,
				decay_steps=1000,
				decay_rate=0.90,
				staircase=True)
			opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
		elif (optimizer == 'sgd_lrs'):
			# --- parameters ---
			#  https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
			#    but, initial learning rate is default of Adam()
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
				0.01,
				decay_steps=1000,
				decay_rate=0.9,
				staircase=True)
			opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
		elif (optimizer == 'momentum'):
			# --- parameters ---
			#  https://qiita.com/8128/items/2d441e46643f73c0ca19
			opt = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-4, momentum=0.9, nesterov=True)
#			opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.8, nesterov=True)
		else:
			print('[ERROR] Unknown optimizer: {}'.format(optimizer))
			quit()
		
		self.model.compile(
			optimizer=opt,
			loss=loss,
			metrics=['accuracy'])
		
		return
	
	# --- 学習 ---
	def fit(self, x_train, y_train,
			x_val=None, y_val=None, x_test=None, y_test=None,
			web_app_ctrl_fifo=None, trainer_ctrl_fifo=None,
			da_params=None,
			batch_size=32, epochs=200,
			verbose=0):
		# --- 学習 ---
		os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
		checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'model.ckpt')
		cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
		es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
		custom_callback = self.CustomCallback(trainer_ctrl_fifo)
		tensorboard_logdir = os.path.join(self.output_dir, 'logs')
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1)
		#callbacks = [cp_callback, es_callback]
		callbacks = [cp_callback, custom_callback, tensorboard_callback]
		
		if (da_params is not None):
			# --- no tuning ---
			datagen = ImageDataGenerator(
				rotation_range=da_params['rotation_range'],
				width_shift_range=da_params['width_shift_range'],
				height_shift_range=da_params['height_shift_range'],
				zoom_range=da_params['zoom_range'],
				channel_shift_range=da_params['channel_shift_range'],
				horizontal_flip=da_params['horizontal_flip'])
		else:
			datagen = ImageDataGenerator()
		datagen.fit(x_train)
		
		if ((x_val is not None) and (y_val is not None)):
			history = self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
						steps_per_epoch=len(x_train)/batch_size, validation_data=(x_val, y_val),
						epochs=epochs, callbacks=callbacks,
						verbose=verbose)
		else:
			history = self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
						steps_per_epoch=len(x_train)/batch_size, validation_split=0.2,
						epochs=epochs, callbacks=callbacks,
						verbose=verbose)
		
		# --- 学習結果を評価 ---
		if ((x_test is not None) and (y_test is not None)):
			test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
			print('Test Accuracy: {}'.format(test_acc))
			print('Test Loss: {}'.format(test_loss))
		
		# --- メトリクスを保存 ---
		metrics = history.history
		os.makedirs(os.path.join(self.output_dir, 'metrics'), exist_ok=True)
		df_metrics = pd.DataFrame(metrics)
		df_metrics.to_csv(os.path.join(self.output_dir, 'metrics', 'metrics.csv'), index_label='epoch')
		
		epoch = df_metrics.index.values
		for column in df_metrics.columns:
			plt.figure()
			plt.plot(epoch, df_metrics[column])
			plt.xlabel('epoch')
			plt.ylabel(column)
			plt.grid(True)
			plt.tight_layout()
			
			graph_name = os.path.join(self.output_dir, 'metrics', '{}.png'.format(column))
			plt.savefig(graph_name)
			
			plt.close()
		
		# --- 学習完了をアプリへ通知 ---
		if (web_app_ctrl_fifo is not None):
			with open(web_app_ctrl_fifo, 'w') as f:
				f.write('trainer_done\n')
		
		return
	
	# --- 推論 ---
	def predict(self, x_test):
		predictions = self.model.predict(x_test)
		return predictions
		
	# --- モデル保存 ---
	def save_model(self):
		# --- 保存先ディレクトリ作成 ---
		model_dir = os.path.join(self.output_dir, 'models')
		os.makedirs(os.path.join(model_dir, 'checkpoint'), exist_ok=True)
		os.makedirs(os.path.join(model_dir, 'saved_model'), exist_ok=True)
		os.makedirs(os.path.join(model_dir, 'hdf5'), exist_ok=True)
		
		# --- checkpoint ---
		self.model.save_weights(os.path.join(model_dir, 'checkpoint', 'model.ckpt'))
		
		# --- saved_model ---
		self.model.save(os.path.join(model_dir, 'saved_model'))
		
		# --- hdf5 ---
		self.model.save(os.path.join(model_dir, 'hdf5', 'model.h5'))
		
		return
	
	# --- メモリリソース解放(セッションのクリア) ---
	def release_memory(self):
		
		keras.backend.clear_session()
		del self.model
		gc.collect()

		return
		
	# --- ラベルインデックス取得 ---
	def GetLabelIndex(self, label, onehot=True):
		if (onehot):
			label = np.argmax(label, axis=1)
		n_category = max(label)+1
		
		return np.array([np.arange(len(label))[label==i] for i in range(n_category)])
	
	# --- システム情報を取得 ---
	def GetSystemInfo():
		_system_info = device_lib.list_local_devices()
		system_info = []
		for info in _system_info:
			dict = {}
			dict['name'] = info.name
			dict['device_type'] = info.device_type
			dict['physical_device_desc'] = info.physical_device_desc
			system_info.append(dict)
		
		return system_info
	
#---------------------------------
# クラス; ResNet学習モジュール
#---------------------------------
class TrainerResNet(Trainer):
	# --- コンストラクタ ---
	def __init__(self, input_shape, classes, output_dir=None, model_file=None, model_type='custom', optimizer='adam', loss='sparse_categorical_crossentropy', initializer='glorot_uniform', dropout_rate=0.0):
		# --- Residual Block ---
		#  * アプリケーションからkeras.applications.resnet.ResNetにアクセスできない為，
		#    必要なモジュールをTensorFlow公式からコピー
		#      https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet.py#L212
		def block1(x, filters, kernel_size=3, stride=1, initializer='glorot_uniform', conv_shortcut=True, name=None):
			bn_axis = 3
			
			if conv_shortcut:
				shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride, kernel_initializer=initializer, name=name + '_0_conv')(x)
				shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
			else:
				shortcut = x

			x = keras.layers.Conv2D(filters, 1, strides=stride, kernel_initializer=initializer, name=name + '_1_conv')(x)
			x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
			x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

			x = keras.layers.Conv2D(filters, kernel_size, padding='SAME', kernel_initializer=initializer, name=name + '_2_conv')(x)
			x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
			x = keras.layers.Activation('relu', name=name + '_2_relu')(x)

			x = keras.layers.Conv2D(4 * filters, 1, kernel_initializer=initializer, name=name + '_3_conv')(x)
			x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

			x = keras.layers.Add(name=name + '_add')([shortcut, x])
			x = keras.layers.Activation('relu', name=name + '_out')(x)
			return x
		
		# --- Residual Block stack ---
		#  * アプリケーションからkeras.applications.resnet.ResNetにアクセスできない為，
		#    必要なモジュールをTensorFlow公式からコピー
		#      https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/applications/resnet.py#L257
		def stack1(x, filters, blocks, stride1=2, dropout_rate=0.0, name=None):
			x = block1(x, filters, stride=stride1, name=name + '_block1')
			x = keras.layers.Dropout(dropout_rate)(x)
			for i in range(2, blocks + 1):
				x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
				x = keras.layers.Dropout(dropout_rate)(x)
			return x
		
		# --- モデル構築 ---
		#  * stack_fn()の関数ポインタを引数に設定してカスタマイズ
		def _load_model(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
			input = keras.layers.Input(shape=input_shape)
			bn_axis = 3
			
			x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input)
			x = keras.layers.Conv2D(64, 7, strides=2, use_bias=True, kernel_initializer=initializer, name='conv1_conv')(x)

			x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
			x = keras.layers.Activation('relu', name='conv1_relu')(x)

			x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
			x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
			x = keras.layers.Dropout(dropout_rate)(x)

			x = stack_fn(x, dropout_rate=dropout_rate)

			x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
			x = keras.layers.Dropout(dropout_rate)(x)
			x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
			
			model = keras.models.Model(input, x)
			model.summary()
			
			return model
			
		# --- モデル構築 ---
		#  * stack_fn()の関数ポインタを引数に設定してカスタマイズ
		def _load_model_deep(input_shape, classes, stack_fn, initializer='glorot_uniform', dropout_rate=0.0):
			input = keras.layers.Input(shape=input_shape)
			bn_axis = 3
			
			x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input)
			x = keras.layers.Conv2D(64, 3, strides=2, use_bias=True, kernel_initializer=initializer, name='conv1_conv')(x)

			x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
			x = keras.layers.Activation('relu', name='conv1_relu')(x)

			x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
			x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
			x = keras.layers.Dropout(dropout_rate)(x)

			x = stack_fn(x, dropout_rate=dropout_rate)

			x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
			x = keras.layers.Dropout(dropout_rate)(x)
			x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
			
			model = keras.models.Model(input, x)
			model.summary()
			
			return model
			
		def _load_model_resnet50(input_shape, classes, initializer='glorot_uniform', dropout_rate=0.0, pretrained=True):
			# --- TensorFlowのResNet50のモデル ---
			#  https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
			#    dbg_mode=0: original ResNet50, dbg_mode=11: custom ResNet50
			if (pretrained):
				print('[INFO] Load ResNet50 model from keras.applications')
				model = keras.applications.resnet50.ResNet50()
			else:
				def stack_fn(x, dropout_rate=0.0):
					x = stack1(x, 64, 3, stride1=1, dropout_rate=dropout_rate, name='conv2')
					x = stack1(x, 128, 4, dropout_rate=dropout_rate, name='conv3')
					x = stack1(x, 256, 6, dropout_rate=dropout_rate, name='conv4')
					return stack1(x, 512, 3, dropout_rate=dropout_rate, name='conv5')
				
				print('[INFO] Load ResNet50 model (custom implementation)')
				model = _load_model(input_shape, classes, stack_fn, initializer=initializer, dropout_rate=dropout_rate)
				
			return model
		
		# --- 基底クラスの初期化 ---
		super().__init__(output_dir=output_dir, model_file=model_file)
		
		# --- モデル構築 ---
		if (self.model is None):
			if (model_type == 'custom'):
				def stack_fn(x, dropout_rate=0.0):
					x = stack1(x, 32, 3, stride1=1, dropout_rate=dropout_rate, name='conv2')
					return stack1(x, 64, 4, dropout_rate=dropout_rate, name='conv3')
				
				self.model = _load_model(input_shape, classes, stack_fn, initializer=initializer, dropout_rate=dropout_rate)
				self._compile_model(optimizer=optimizer, loss=loss)
			elif (model_type == 'custom_deep'):
				def stack_fn(x, dropout_rate=0.0):
					x = stack1(x, 16, 18, stride1=1, dropout_rate=dropout_rate, name='conv2')
					x = stack1(x, 32, 18, dropout_rate=dropout_rate, name='conv3')
					return stack1(x, 64, 18, dropout_rate=dropout_rate, name='conv4')
				
				self.model = _load_model_deep(input_shape, classes, stack_fn, initializer=initializer, dropout_rate=dropout_rate)
				self._compile_model(optimizer=optimizer, loss=loss)
			elif (model_type == 'resnet50'):
				self.model = _load_model_resnet50(input_shape, classes, initializer=initializer, dropout_rate=dropout_rate, pretrained=False)
				self._compile_model(optimizer=optimizer, loss=loss)
			else:
				print('[ERROR] Unknown model_type: {}'.format(model_type))
				return
			
		if (self.output_dir is not None):
			keras.utils.plot_model(self.model, os.path.join(self.output_dir, 'plot_model.png'), show_shapes=True)
		
		return
	
#---------------------------------
# クラス; CNN学習モジュール
#---------------------------------
class TrainerCNN(Trainer):
	# --- コンストラクタ ---
	def __init__(self, input_shape, output_dir=None, model_file=None, optimizer='adam', loss='sparse_categorical_crossentropy', initializer='glorot_uniform', model_type='baseline'):
		# --- モデル構築(baseline) ---
		def _load_model(input_shape, initializer='glorot_uniform'):
			model = keras.models.Sequential()
			model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer=initializer))
			model.add(keras.layers.MaxPooling2D((2, 2)))
			model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
			model.add(keras.layers.MaxPooling2D((2, 2)))
			model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
			model.add(keras.layers.MaxPooling2D((2, 2)))
			model.add(keras.layers.Flatten(input_shape=input_shape))
			model.add(keras.layers.Dense(64, activation='relu'))
			model.add(keras.layers.Dense(10, activation='softmax'))
			
			model.summary()
			
			return model
		
		# --- モデル構築(deep_model) ---
		def _load_model_deep(input_shape, initializer='glorot_uniform'):
			input = keras.layers.Input(shape=input_shape)

			x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation="relu")(input)
			x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.BatchNormalization()(x)
			x = keras.layers.Conv2D(64, (3,3), padding="SAME", activation= "relu")(x)
			x = keras.layers.MaxPooling2D()(x)
			x = keras.layers.Dropout(0.25)(x)

			x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.BatchNormalization()(x)
			x = keras.layers.Conv2D(128, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.MaxPooling2D()(x)
			x = keras.layers.Dropout(0.25)(x)

			x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.BatchNormalization()(x)
			x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.Conv2D(256, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.BatchNormalization()(x)
			x = keras.layers.Conv2D(512, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.Conv2D(512, (3,3), padding="SAME", activation="relu")(x)
			x = keras.layers.GlobalAveragePooling2D()(x)

			x = keras.layers.Dense(1024, activation="relu")(x)
			x = keras.layers.Dropout(0.5)(x)
			x = keras.layers.Dense(1024, activation="relu")(x)
			x = keras.layers.Dropout(0.5)(x)
			x = keras.layers.Dense(10, activation="softmax")(x)
			
			model = keras.models.Model(input, x)
			model.summary()
			
			return model
		
		# --- 基底クラスの初期化 ---
		super().__init__(output_dir=output_dir, model_file=model_file)
		
		# --- モデル構築 ---
		if (self.model is None):
			if (model_type == 'baseline'):
				self.model = _load_model(input_shape, initializer=initializer)
			elif (model_type == 'deep_model'):
				self.model = _load_model_deep(input_shape, initializer=initializer)
			else:
				print('[ERROR] Unknown model_type: {}'.format(model_type))
				quit()
		
		self._compile_model(optimizer=optimizer, loss=loss)
		if (self.output_dir is not None):
			keras.utils.plot_model(self.model, os.path.join(self.output_dir, 'plot_model.png'), show_shapes=True)
		
		return
	

#---------------------------------
# クラス; MLP学習モジュール
#---------------------------------
class TrainerMLP(Trainer):
	# --- コンストラクタ ---
	def __init__(self, input_shape, output_dir=None, model_file=None, optimizer='adam'):
		# --- モデル構築 ---
		def _load_model(input_shape):
			model = keras.models.Sequential()
			model.add(keras.layers.Flatten(input_shape=input_shape))
			model.add(keras.layers.Dense(128, activation='relu'))
			model.add(keras.layers.Dense(10, activation='softmax'))
			
			model.summary()
			
			return model
		
		# --- 基底クラスの初期化 ---
		super().__init__(output_dir=output_dir, model_file=model_file)
		
		# --- モデル構築 ---
		if (self.model is None):
			self.model = _load_model(input_shape)
			self._compile_model(optimizer=optimizer)
			if (self.output_dir is not None):
				keras.utils.plot_model(self.model, os.path.join(self.output_dir, 'plot_model.png'), show_shapes=True)
		
		return
	


#---------------------------------
# メイン処理; Trainerモジュールテスト
#---------------------------------
def main():
	import argparse
	def _argparse():
		parser = argparse.ArgumentParser(description='Trainerモジュールテスト\n'
					'  * test_mode=\'ResNet\': ResNetのモデル構造確認(ResNet50の構造をTensorFlow公開モデルと比較)',
					formatter_class=argparse.RawTextHelpFormatter)

		# --- 引数を追加 ---
		parser.add_argument('--test_mode', dest='test_mode', type=str, default='ResNet', required=False, \
				help='テストモード(ResNet)')

		args = parser.parse_args()
		return args

	# --- 引数処理 ---
	args = _argparse()
	print(args.test_mode)
	
	# --- モジュールテスト ---
	if (args.test_mode == 'ResNet'):
		trainer = TrainerResNet([224, 224, 3], 1000, output_dir=None, model_type='resnet50')
	else:
		print('[ERROR] Unknown test_mode: {}'.format(args.test_mode))
	
	return

	
if __name__ == '__main__':
	main()
