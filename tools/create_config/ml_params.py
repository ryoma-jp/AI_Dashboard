#! -*- coding: utf-8 -*-

class MlParams():
	"""Class MlParams
	
	* 環境設定や学習パラメータの基底クラス(パラメータテンプレート)
	* 各パラメータは下記のキーで構成する
	  * description: パラメータの説明
	  * value: パラメータの値
	  * configurable: ブラウザ上での設定変更可否(True：変更可，False:変更不可)
	  * selectable: パラメータ変更時の選択可否(True：ドロップダウンで変更，False:テキストボックスで変更)
	                configurable=Trueのときのみ有効
	  * items: ドロップダウンで選択するアイテムのリスト
	            selectable=Trueのときのみ有効
	"""
	
	def __init__(self):
		self.params = {
			'env': {
				'web_app_ctrl_fifo': {
					'name': 'Web Application Control FIFO',
					'description': 'Web application control FIFO(Trainer → Web App)',
					'dtype': 'str',
					'value': None,
					'configurable': False,
				},
				'trainer_ctrl_fifo': {
					'name': 'Trainer Control FIFO',
					'description': 'Trainer control FIFO(Web App → Trainer)',
					'dtype': 'str',
					'value': None,
					'configurable': False,
				},
				'result_dir': {
					'name': 'Result Directory',
					'description': 'Directory into training result files(model, log, and etc)',
					'dtype': 'str',
					'value': None,
					'configurable': False,
				},
				'tensorboard_port': {
					'name': 'Tensorboard PORT',
					'description': 'PORT using Tensorboard',
					'dtype': 'int',
					'value': 6006,
					'configurable': False,
				},
			},
			'dataset': {
				'dataset_name': {
					'name': 'Dataset Name',
					'description': 'Dataset name',
					'dtype': 'str',
					'value': None,
					'configurable': False,
				},
				'dataset_dir': {
					'name': 'Dataset Directory',
					'description': 'Dataset directory',
					'dtype': 'str',
					'value': None,
					'configurable': False,
				},
				'norm': {
					'name': 'Normalization',
					'description': 'Normalization method',
					'dtype': 'str',
					'value': 'z-score',
					'configurable': True,
					'selectable': True,
					'items': ['max', 'max-min', 'z-score'],
				},
				'image_data_augmentation': {
					'rotation_range': {
						'name': 'Rotation Range',
						'description': 'Rotation range',
						'dtype': 'int',
						'value': 5,
						'configurable': True,
						'selectable': False,
					},
					'width_shift_range': {
						'name': 'Width Shift Range',
						'description': 'Width range for horizontal shift',
						'dtype': 'float',
						'value': 0.2,
						'configurable': True,
						'selectable': False,
					},
					'height_shift_range': {
						'name': 'Height Shift Range',
						'description': 'Height range for vertical shift',
						'dtype': 'float',
						'value': 0.2,
						'configurable': True,
						'selectable': False,
					},
					'zoom_range': {
						'name': 'Zoom Range',
						'description': 'Zoom range',
						'dtype': 'float',
						'value': 0.2,
						'configurable': True,
						'selectable': False,
					},
					'channel_shift_range': {
						'name': 'Channel Shift Range',
						'description': 'Channel shift range',
						'dtype': 'float',
						'value': 0.2,
						'configurable': True,
						'selectable': False,
					},
					'horizontal_flip': {
						'name': 'Horizontal Flip',
						'description': 'Enable horizontal flip',
						'dtype': 'bool',
						'value': True,
						'configurable': True,
						'selectable': True,
						'items': [True, False],
					},
				},
			},
			'model': {
				'model_type': {
					'name': 'Model Type',
					'description': 'Model Structure',
					'dtype': 'str',
					'value': 'SimpleCNN',
					'configurable': True,
					'selectable': True,
					'items': ['MLP', 'SimpleCNN', 'DeepCNN', 'SimpleResNet', 'DeepResNet'],
				},
			},
			'training_parameter': {
				'optimizer': {
					'name': 'Optimizer',
					'description': 'Optimizer',
					'dtype': 'str',
					'value': 'momentum',
					'configurable': True,
					'selectable': True,
					'items': ['momentum', 'adam', 'sgd', 'adam_lrs', 'sgd_lrs'],
				},
				'batch_size': {
					'name': 'Batch Size',
					'description': 'Batch size',
					'dtype': 'int',
					'value': 100,
					'configurable': True,
					'selectable': False,
				},
				'initializer': {
					'name': 'Initializer',
					'description': 'Weight initializer',
					'dtype': 'str',
					'value': 'he_normal',
					'configurable': True,
					'selectable': True,
					'items': ['glrot_uniform', 'glrot_normal', 'he_uniform', 'he_normal'],
				},
				'dropout_rate': {
					'name': 'Dropout Rate',
					'description': 'Dropout rate',
					'dtype': 'float',
					'value': 0.25,
					'configurable': True,
					'selectable': False,
				},
				'loss_func': {
					'name': 'Loss Function',
					'description': 'Loss Function',
					'dtype': 'str',
					'value': 'categorical_crossentropy',
					'configurable': True,
					'selectable': True,
					'items': ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy'],
				},
				'epochs': {
					'name': 'EPOCHs',
					'description': 'Epochs',
					'dtype': 'int',
					'value': 400,
					'configurable': True,
					'selectable': False,
				},
			},
		}

class MlParams_MNIST(MlParams):
	"""Class MlParams_MNIST
	
	* MNISTデータセットによる識別モデル学習時のパラメータ
	"""
	
	def __init__(self):
		super().__init__()
		self.params['dataset']['dataset_name']['value'] = 'MNIST'
		self.params['dataset']['norm']['value'] = 'max'
		self.params['dataset']['image_data_augmentation']['horizontal_flip']['value'] = False

class MlParams_CIFAR10(MlParams):
	"""Class MlParams_CIFAR10
	
	* CIFAR-10データセットによる識別モデル学習時のパラメータ
	"""
	
	def __init__(self):
		super().__init__()
		self.params['dataset']['dataset_name']['value'] = 'CIFAR-10'

