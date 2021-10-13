#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import numpy as np

#---------------------------------
# クラス; データ取得基底クラス
#---------------------------------
class DataLoader():
	# --- コンストラクタ ---
	def __init__(self):
		return
	
	# --- 標準化・正規化 ---
	#  * mode: 正規化手法
	#     max: データの絶対値の最大が1.0となるように正規化(最大値で除算)
	#     max-min: データの最大が1.0最小が0.0となるように正規化(最大値と最小値を用いて算出)
	#     z-score: 標準化(平均と標準偏差を用いて算出)
	#
	#  [Reference]
	#  * 標準化と正規化: https://aiacademy.jp/texts/show/?id=555
	def normalization(self, mode):
		if (mode == 'max'):
			train_norm = self.train_images / 255.
			validation_norm = self.validation_images / 255.
			test_norm = self.test_images / 255.
		elif (mode == 'max-min'):
			train_min = np.min(self.train_images)
			train_diff = np.max(self.train_images) - np.min(self.train_images)
			
			train_norm = (self.train_images - train_min) / train_diff
			validation_norm = (self.validation_images - train_min) / train_diff
			test_norm = (self.test_images - train_min) / train_diff
		elif (mode == 'z-score'):
			train_mean = np.mean(self.train_images)
			train_std = np.std(self.train_images)
			
			train_norm = (self.train_images - train_mean) / train_std
			validation_norm = (self.validation_images - train_mean) / train_std
			test_norm = (self.test_images - train_mean) / train_std
		else:
			print('[ERROR] Unknown data normalization mode: {}'.format(mode))
			quit()
		
		return train_norm, validation_norm, test_norm
		
	# --- ラベルインデックス取得 ---
	def get_label_index(self, label, one_hot=True):
		if (one_hot):
			label = np.argmax(label, axis=1)
		n_category = max(label)+1
		
		return np.array([np.arange(len(label))[label==i] for i in range(n_category)])
	
	# --- 学習データとバリデーションデータを分割 ---
	def split_train_val(self, validation_split):
		idx = np.arange(len(self.train_images))
		np.random.shuffle(idx)
		
		if ((1.0 - validation_split) == 1.0):
			self.validation_images = None
			self.validation_labels = None
		elif ((1.0 - validation_split) == 0.0):
			self.validation_images = self.train_images
			self.validation_labels = self.train_labels
			self.train_images = None
			self.train_labels = None
		else:
			validation_index = int(len(self.train_images) * (1.0-validation_split))
			self.validation_images = self.train_images[validation_index:]
			self.validation_labels = self.train_labels[validation_index:]
			self.train_images = self.train_images[0:validation_index]
			self.train_labels = self.train_labels[0:validation_index]
		
		return

#---------------------------------
# クラス; CIFAR-10データセット取得
#---------------------------------
class DataLoaderCIFAR10(DataLoader):
	def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False):
		"""
			[引数説明]
				* validation_split: validation dataとして使用する学習データの比率(0.0 ～ 1.0)
				* flatten: 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
				* one_hot: one hot形式(=True)かラベルインデックス(=False;default)かを選択する
		"""
		
		def unpickle(file):
			import pickle
			with open(file, 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
			return dict
		
		# --- initialize super class ---
		super().__init__()

		# --- load training data ---
		train_data_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
		dict_data = unpickle(os.path.join(dataset_dir, train_data_list[0]))
		train_images = dict_data[b'data']
		train_labels = dict_data[b'labels'].copy()
		for train_data in train_data_list[1:]:
			dict_data = unpickle(os.path.join(dataset_dir, train_data))
			train_images = np.vstack((train_images, dict_data[b'data']))
			train_labels = np.hstack((train_labels, dict_data[b'labels']))
		
		# --- load test data ---
		test_data = "test_batch"
		dict_data = unpickle(os.path.join(dataset_dir, test_data))
		test_images = dict_data[b'data']
		test_labels = dict_data[b'labels'].copy()
		
		# --- transpose: [N, C, H, W] -> [N, H, W, C] ---
		self.train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		self.test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
		
		# --- labels ---
		if (one_hot):
			identity = np.eye(10, dtype=np.int)
			self.train_labels = np.array([identity[i] for i in train_labels])
			self.test_labels = np.array([identity[i] for i in test_labels])
		else:
			self.train_labels = np.array(train_labels)
			self.test_labels = np.array(test_labels)
		
		# --- 学習データとバリデーションデータを分割 ---
		self.split_train_val(validation_split)
		
		# --- 出力次元数を保持 ---
		self.output_dims = 10
		
		return
		
#---------------------------------
# クラス; MNISTデータセット取得
#---------------------------------
class DataLoaderMNIST(DataLoader):
	# --- コンストラクタ ---
	def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False):
		"""
			[引数説明]
				* validation_split: validation dataとして使用する学習データの比率(0.0 ～ 1.0)
				* flatten: 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
				* one_hot: one hot形式(=True)かラベルインデックス(=False;default)かを選択する
		"""
		
		# --- initialize super class ---
		super().__init__()
		
		# --- load training data ---
		f = open(os.path.join(dataset_dir, 'train-images-idx3-ubyte'))
		byte_data = np.fromfile(f, dtype=np.uint8)
		
		n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
		img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
		img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
		
		if (flatten):
			self.train_images = byte_data[16:].reshape(n_items, -1)
		else:
			self.train_images = byte_data[16:].reshape(n_items, img_h, img_w, 1)
		
		# --- load training label ---
		f = open(os.path.join(dataset_dir, 'train-labels-idx1-ubyte'))
		byte_data = np.fromfile(f, dtype=np.uint8)
		
		n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
		
		self.train_labels = byte_data[8:]
		if (one_hot):
			identity = np.eye(10, dtype=np.int)
			self.train_labels = np.array([identity[i] for i in self.train_labels])
		
		# --- load test data ---
		f = open(os.path.join(dataset_dir, 't10k-images-idx3-ubyte'))
		byte_data = np.fromfile(f, dtype=np.uint8)
		
		n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
		img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
		img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
		
		if (flatten):
			self.test_images = byte_data[16:].reshape(n_items, -1)
		else:
			self.test_images = byte_data[16:].reshape(n_items, img_h, img_w, 1)
		
		# --- load test label ---
		f = open(os.path.join(dataset_dir, 't10k-labels-idx1-ubyte'))
		byte_data = np.fromfile(f, dtype=np.uint8)
		
		n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
		
		self.test_labels = byte_data[8:]
		if (one_hot):
			self.test_labels = np.array([identity[i] for i in self.test_labels])
		
		# --- 学習データとバリデーションデータを分割 ---
		self.split_train_val(validation_split)
		
		# --- 出力次元数を保持 ---
		self.output_dims = 10
		
		return
	
