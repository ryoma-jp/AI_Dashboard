#! -*- coding: utf-8 -*-

#---------------------------------
# クラス: MlParams
#   * 学習パラメータの基底クラス(パラメータテンプレート)
#---------------------------------
class MlParams():
	def __init__(self):
		self.params = {
			'env': {
				'fifo': '/tmp/fifo_trainer_ctl',
				'result_dir': None,
				'tensorboard_port': 6006,
			},
			'dataset': {
				'data_type': None,
				'dataset_dir': None,
				'data_augmentation': {
					'rotation_range': 5,
					'width_shift_range': 0.2,
					'height_shift_range': 0.2,
					'zoom_range': 0.2,
					'channel_shift_range': 0.2,
					'horizontal_flip': True
				},
			},
			'model': {
				'model_type': 'SimpleCNN',
			},
			'hyper_parameters': {
				'optimizer': 'momentum',
				'batch_size': 100,
				'initializer': 'he_normal',
				'dropout_rate': 0.25,
				'loss_func': 'categorical_crossentropy',
				'epochs': 400,
			},
		}

#---------------------------------
# クラス: MlParams_MNIST
#   * MNISTデータセットによる識別モデル学習時のパラメータ
#---------------------------------
class MlParams_MNIST(MlParams):
	def __init__(self):
		super().__init__()
		self.params['dataset']['data_type'] = 'MNIST'

#---------------------------------
# クラス: MlParams_CIFAR10
#   * CIFAR10データセットによる識別モデル学習時のパラメータ
#---------------------------------
class MlParams_CIFAR10(MlParams):
	def __init__(self):
		super().__init__()
		self.params['dataset']['data_type'] = 'CIFAR-10'

