#! -*- coding: utf-8 -*-

#---------------------------------
# クラス: MlParams
#   * 学習パラメータの基底クラス(パラメータテンプレート)
#---------------------------------
class MlParams():
	def __init__(self):
		self.params = {
			'env': {
				'fifo': {
					'description': 'Trainer control FIFO',
					'value': '/tmp/fifo_trainer_ctl',
				},
				'result_dir': {
					'description': 'Directory into training result files(model, log, and etc)',
					'value': None,
				},
				'tensorboard_port': {
					'description': 'PORT using Tensorboard',
					'value': 6006,
				},
			},
			'dataset': {
				'dataset_name': {
					'description': 'Dataset name',
					'value': None,
				},
				'dataset_dir': {
					'description': 'Dataset directory',
					'value': None,
				},
				'data_augmentation': {
					'rotation_range': {
						'description': 'Rotation range',
						'value': 5,
					},
					'hshift_range': {
						'description': 'Horizontal shift range',
						'value': 0.2,
					},
					'vshift_range': {
						'description': 'Vertical shift range',
						'value': 0.2,
					},
					'zoom_range': {
						'description': 'Zoom range',
						'value': 0.2,
					},
					'channel_shift_range': {
						'description': 'Channel shift range',
						'value': 0.2,
					},
					'horizontal_flip': {
						'description': 'Enable horizontal flip',
						'value': True,
					},
				},
			},
			'model': {
				'model_type': {
					'description': 'Model Structure',
					'value': 'SimpleCNN',
				},
			},
			'training_parameter': {
				'optimizer': {
					'description': 'Optimizer',
					'value': 'momentum',
				},
				'batch_size': {
					'description': 'Batch size',
					'value': 100,
				},
				'initializer': {
					'description': 'Weight initializer',
					'value': 'he_normal',
				},
				'dropout_rate': {
					'description': 'Dropout rate',
					'value': 0.25,
				},
				'loss_func': {
					'description': 'Loss Function',
					'value': 'categorical_crossentropy',
				},
				'epochs': {
					'description': 'Epochs',
					'value': 400,
				},
			},
		}

#---------------------------------
# クラス: MlParams_MNIST
#   * MNISTデータセットによる識別モデル学習時のパラメータ
#---------------------------------
class MlParams_MNIST(MlParams):
	def __init__(self):
		super().__init__()
		self.params['dataset']['dataset_name']['value'] = 'MNIST'

#---------------------------------
# クラス: MlParams_CIFAR10
#   * CIFAR10データセットによる識別モデル学習時のパラメータ
#---------------------------------
class MlParams_CIFAR10(MlParams):
	def __init__(self):
		super().__init__()
		self.params['dataset']['dataset_name']['value'] = 'CIFAR-10'

