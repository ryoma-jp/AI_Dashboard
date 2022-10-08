#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import logging
import numpy as np
import pandas as pd
import requests
import tarfile
import gzip
import cv2

#---------------------------------
# クラス; データ取得基底クラス
#---------------------------------
class DataLoader():
	# --- コンストラクタ ---
	def __init__(self):
		self.one_hot = True
		self.output_dims = -1
		
		return
	
	# --- ファイルダウンロード ---
	def file_download(self, dir, url):
		save_file = os.path.join(dir, os.path.basename(url))
		content = requests.get(url).content
		
		with open(save_file, mode='wb') as f:
			f.write(content)
		
		return save_file
	
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
			logging.debug('[ERROR] Unknown data normalization mode: {}'.format(mode))
			quit()
		
		return train_norm, validation_norm, test_norm
		
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
	
	def convert_label_encoding(self, one_hot=True):
		"""Convert Label Encoding
		
		正解ラベルをOne Hot表現とインデックス表現を相互変換する
		
		Args:
		    one_hot (bool): 変換先のインデックス表現を指定
		"""
		
		if ((not self.one_hot) and (one_hot)):
			identity = np.eye(self.output_dims, dtype=np.int)
			self.train_labels = np.array([identity[i] for i in self.train_labels])
			self.validation_labels = np.array([identity[i] for i in self.validation_labels])
			self.test_labels = np.array([identity[i] for i in self.test_labels])
		elif ((self.one_hot) and (not one_hot)):
			self.train_labels = self.train_labels.argmax(axis=1)
			self.validation_labels = self.validation_labels.argmax(axis=1)
			self.test_labels = self.test_labels.argmax(axis=1)
			
		self.one_hot = one_hot
		
		return
	
#---------------------------------
# クラス; CIFAR-10データセット取得
#---------------------------------
class DataLoaderCIFAR10(DataLoader):
	def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False):
		"""
			[引数説明]
				* validation_split: validation dataとして使用する学習データの比率(0.0 ～ 1.0)
				* flatten: 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
				* one_hot: one hot形式(=True)かラベルインデックス(=False;default)かを選択する
				* download: データをダウンロードする場合にTrueを指定
		"""
		
		def unpickle(file):
			import pickle
			with open(file, 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
			return dict
		
		# --- initialize super class ---
		super().__init__()
		self.one_hot = one_hot
		
		# --- download dataset and extract ---
		if (download):
			logging.debug('[DataLoaderCIFAR10] {}'.format(dataset_dir))
			os.makedirs(dataset_dir, exist_ok=True)
			if (not os.path.exists(os.path.join(dataset_dir, 'cifar-10-batches-py'))):
				url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
				save_file = self.file_download(dataset_dir, url)
				
				with tarfile.open(save_file) as tar:
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(tar, path=dataset_dir)
			else:
				logging.debug('CIFAR-10 dataset is exists (Skip Download)')
		dataset_dir = os.path.join(dataset_dir, 'cifar-10-batches-py')
			
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
		if (self.one_hot):
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
	def __init__(self, dataset_dir, validation_split=0.0, flatten=False, one_hot=False, download=False):
		"""
			[引数説明]
				* validation_split: validation dataとして使用する学習データの比率(0.0 ～ 1.0)
				* flatten: 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
				* one_hot: one hot形式(=True)かラベルインデックス(=False;default)かを選択する
		"""
		
		# --- initialize super class ---
		super().__init__()
		self.one_hot = one_hot
		
		# --- download dataset and extract ---
		if (download):
			logging.debug('[DataLoaderMNIST] {}'.format(dataset_dir))
			os.makedirs(dataset_dir, exist_ok=True)
			mnist_files = [
				'train-images-idx3-ubyte.gz',
				'train-labels-idx1-ubyte.gz',
				't10k-images-idx3-ubyte.gz',
				't10k-labels-idx1-ubyte.gz'
			]
			
			for mnist_file in mnist_files:
				if (not os.path.exists(os.path.join(dataset_dir, mnist_file))):
					url = 'http://yann.lecun.com/exdb/mnist/' + mnist_file
					save_file = self.file_download(dataset_dir, url)
					
					with gzip.open(save_file, 'rb') as gz:
						gz_content = gz.read()
					
					save_file = os.path.join(dataset_dir, mnist_file[:-3])
					with open(save_file, 'wb') as f:
						f.write(gz_content)
					
				else:
					logging.debug('{} is exists (Skip Download)'.format(mnist_file))
			
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
		if (self.one_hot):
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
		if (self.one_hot):
			self.test_labels = np.array([identity[i] for i in self.test_labels])
		
		# --- 学習データとバリデーションデータを分割 ---
		self.split_train_val(validation_split)
		
		# --- 出力次元数を保持 ---
		self.output_dims = 10
		
		return
	

#---------------------------------
# クラス; カスタムデータセット取得
#---------------------------------
class DataLoaderCustom(DataLoader):
	# --- コンストラクタ ---
	def __init__(self, train_dir, test_dir, validation_dir=None, validation_split=0.0, flatten=False, one_hot=False, download=False):
		"""
			[引数説明]
				* train_dir: 学習データセットのディレクトリ
				* test_dir: テストデータセットのディレクトリ
				* validation_dir: バリデーションデータセットのディレクトリ
				* validation_split: validation dataとして使用する学習データの比率(0.0 ～ 1.0)
				                    validation_dirが指定されている場合は，validation_splitは無視する
				* flatten: 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
				* one_hot: one hot形式(=True)かラベルインデックス(=False;default)かを選択する
		"""
		
		def _load_data(data_dir):
			"""_load_data
				カスタムデータセットを読み込み，画像とラベルを返す
				
				[引数説明]
				  * data_dir: カスタムデータセットのディレクトリ
			"""
			json_file = os.path.join(data_dir, 'info.json')
			df_data = pd.read_json(json_file, orient='records')
			
			img = cv2.imread(os.path.join(data_dir, df_data['file'][0]))
			n_items = len(df_data)
			if (img.ndim) == 3:
				img_h, img_w, n_channel = img.shape
			
			images = []
			labels = []
			for data_ in df_data.itertuples():
				images.append(list(cv2.imread(os.path.join(data_dir, data_.file))))
				labels.append(data_.class_id)
			
			return np.array(images), np.array(labels)
		
		# --- initialize super class ---
		super().__init__()
		self.one_hot = one_hot
		
		# --- load training data ---
		self.train_images, self.train_labels = _load_data(train_dir)
		
		# --- load test data ---
		self.test_images, self.test_labels = _load_data(test_dir)
		
		# --- load validation data ---
		if (validation_dir is not None):
			self.validation_images, self.validation_labels = _load_data(validation_dir)
		else:
			self.split_train_val(validation_split)
			
		# --- 出力次元数を保持 ---
		self.output_dims = 10	# T.B.D
		
		return
	
