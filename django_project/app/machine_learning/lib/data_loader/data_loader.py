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

from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image

from machine_learning.lib.utils.preprocessor import image_preprocess

#---------------------------------
# クラス; データ取得基底クラス
#---------------------------------
class DataLoader():
    """DataLoader
    
    データ取得処理の基底クラス
    
    Attributes:
        one_hot (bool): 真値がonehot表現の場合にTrueにセット
        output_dims (int): 出力次元数
        verified (bool): Webアプリでの解析可否の検証結果(Webアプリで解析可能な場合にTrueにセット)
        train_x (numpy.ndarray): 学習用画像
        train_y (numpy.ndarray): 学習用画像の真値
        validation_x (numpy.ndarray): Validation用画像
        validation_y (numpy.ndarray): Validation用画像の真値
        test_x (numpy.ndarray): Test用画像
        test_y (numpy.ndarray): Test用画像の真値
        dataset_type (str): データセット種別
            - 'img_clf': 画像データの分類タスク
            - 'img_reg': 画像データの回帰タスク
            - 'table_clf': テーブルデータの分類タスク
            - 'table_reg': テーブルデータの回帰タスク
        input_distributions (dict): input feature分布
        target_distributions (dict): target分布
    
    """
    # --- コンストラクタ ---
    def __init__(self):
        self.one_hot = True
        self.output_dims = -1
        self.verified = False
        self.dataset_type = None
        self.input_distributions = None
        self.target_distributions = None
        self.statistic_keys = []
        
    
    # --- ファイルダウンロード ---
    def file_download(self, dir, url):
        """file_download
        
        指定URLのデータをダウンロードする
        
        Args:
            dir (str): ダウンロードデータの保存先のディレクトリ
            url (str): ダウンロードデータのURL
        
        """
        save_file = Path(dir, Path(url).name)
        content = requests.get(url).content
        
        with open(save_file, mode='wb') as f:
            f.write(content)
        
        return save_file
    
    def preprocessing(self, norm_mode='none'):
        """Pre-processing
        
        Data pre-processing.
        Pre-processing methods are below.
            - normalization (or standardization)
        
        Args:
            norm_mode (str): Specify the normalization method
                - 'none': Trough
                - 'max': Divide max value(=255)
                - 'max-min': Normalization to value range that is from 0.0 to 1.0
                - 'z-score': Standardization
        
        Returns:
            train, validation and test dataset are preprocessed.
            return the ``y`` value is for future update, example for add scaling or cropping.
        
        Note:
            - [Standardization and Normalization](https://aiacademy.jp/texts/show/?id=555)
        """
        # --- Initialize ---
        preprocessed_train_x = None
        preprocessed_train_y = self.train_y
        preprocessed_validation_x = None
        preprocessed_validation_y = self.validation_y
        preprocessed_test_x = None
        preprocessed_test_y = self.test_y
        
        # --- Normalization process ---
        if (norm_mode == 'none'):
            norm_coef = [0.0, 1.0]
        elif (norm_mode == 'max'):
            norm_coef = [0.0, 255.0]
        elif (norm_mode == 'max-min'):
            train_min = np.min(self.train_x)
            train_diff = np.max(self.train_x) - np.min(self.train_x)
            train_diff = np.clip(train_diff, np.finfo(float).eps, None)
            
            norm_coef = [train_min, train_diff]
        elif (norm_mode == 'z-score'):
            norm_coef = [np.mean(self.train_x), np.std(self.train_x)]
        else:
            logging.debug('[WARNING] Unknown data normalization mode: {}'.format(mode))
            norm_coef = [0.0, 1.0]
        
        preprocessed_train_x = image_preprocess(self.train_x, norm_coef)
        if (self.validation_x is not None):
            preprocessed_validation_x = image_preprocess(self.validation_x, norm_coef)
        if (self.test_x is not None):
            preprocessed_test_x = image_preprocess(self.test_x, norm_coef)
        
        return preprocessed_train_x, preprocessed_train_y, \
               preprocessed_validation_x, preprocessed_validation_y, \
               preprocessed_test_x, preprocessed_test_y
        
    # --- 学習データとバリデーションデータを分割 ---
    def split_train_val(self, validation_split):
        """split_train_val
        
        学習データとValidationデータを分割する
        
        Args:
            validation_split (float): Validationデータの比率
        
        """
        idx = np.arange(len(self.train_x))
        np.random.shuffle(idx)
        
        if ((1.0 - validation_split) == 1.0):
            self.validation_x = None
            self.validation_y = None
        elif ((1.0 - validation_split) == 0.0):
            self.validation_x = self.train_x
            self.validation_y = self.train_y
            self.train_x = None
            self.train_y = None
        else:
            validation_index = int(len(self.train_x) * (1.0-validation_split))
            self.validation_x = self.train_x[validation_index:]
            self.validation_y = self.train_y[validation_index:]
            self.train_x = self.train_x[0:validation_index]
            self.train_y = self.train_y[0:validation_index]
        
        return
    
    def train_valid_test_split(self, validation_size=0.2, test_size=0.3):
        """Split Dataset
        
        This function split dataset from train to train, validation and test.
        
        Args:
            validation_size (float): validation data rate (0.0 <= validation_size < 1.0)
            test_size (float): validation data rate (0.0 <= validation_size < 1.0)
        """
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y, test_size=test_size, random_state=42)
        
        self.train_x, self.validation_x, self.train_y, self.validation_y = train_test_split(
            self.train_x, self.train_y, test_size=validation_size, random_state=42)
        
    
    def convert_label_encoding(self, one_hot=True):
        """Convert Label Encoding
        
        正解ラベルをOne Hot表現とインデックス表現を相互変換する
        
        Args:
            one_hot (bool): 変換先のインデックス表現を指定
        """
        
        if ((not self.one_hot) and (one_hot)):
            identity = np.eye(self.output_dims, dtype=np.int)
            self.train_y = np.array([identity[i] for i in self.train_y])
            
            if (self.validation_y is not None):
                self.validation_y = np.array([identity[i] for i in self.validation_y])
            
            if (self.test_y is not None):
                self.test_y = np.array([identity[i] for i in self.test_y])
        elif ((self.one_hot) and (not one_hot)):
            self.train_y = self.train_y.argmax(axis=1)
            
            if (self.validation_y is not None):
                self.validation_y = self.validation_y.argmax(axis=1)
            
            if (self.test_y is not None):
                self.test_y = self.test_y.argmax(axis=1)
            
        self.one_hot = one_hot
        
        return
    
    def data_analysis(self):
        """Data Analysis
        
        This function analylizies dataset.
        """
        
        # --- Calculate input distribution ---
        if (self.dataset_type in ['img_clf', 'img_reg']):
            # --- Image data ---
            self.input_distributions = None
        elif (self.dataset_type in ['table_clf', 'table_reg']):
            # --- Table data ---
            
            # --- Initialize ---
            self.input_distributions = {}
            self.statistic_keys.append("Input Distributions")
            
            # --- Set bins
            bins = 20
            
            # --- Calculate input distribution ---
            if (self.train_x is not None):
                self.input_distributions['train'] = {}
                
                for _key in self.train_x.columns:
                    self.input_distributions['train'][_key] = {}
                    hist_y, hist_x = np.histogram(self.train_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['train'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['train'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['train'] = None
                
            if (self.validation_x is not None):
                self.input_distributions['validation'] = {}
                
                for _key in self.validation_x.columns:
                    self.input_distributions['validation'][_key] = {}
                    hist_y, hist_x = np.histogram(self.validation_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['validation'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['validation'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['validation'] = None
                
            if (self.test_x is not None):
                self.input_distributions['test'] = {}
                
                for _key in self.test_x.columns:
                    self.input_distributions['test'][_key] = {}
                    hist_y, hist_x = np.histogram(self.test_x[_key], bins=bins)
                    hist_x = np.round(hist_x, decimals=2)
                    self.input_distributions['test'][_key]['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                    self.input_distributions['test'][_key]['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.input_distributions['test'] = None
                
            
        else:
            # --- Unknown dataset type ---
            self.input_distributions = None
        
        # --- Calculate target distribution ---
        if (self.target_distributions is None):
            # --- Initialize ---
            self.target_distributions = {}
            self.statistic_keys.append("Target Distributions")
            
            # --- Set bins ---
            if (self.dataset_type in ['img_clf', 'table_clf']):
                # --- Set sequence of scalars to bins if dataset is the classification task ---
                bins = np.unique(self.train_y)
                if (self.validation_y is not None):
                    _targets = np.unique(self.validation_y)
                    _targets = np.hstack((_targets, max(_targets)+1))
                    if (len(bins) < len(_targets)):
                        bins = _targets
                if (self.test_y is not None):
                    _targets = np.unique(self.test_y)
                    if (len(bins) < len(_targets)):
                        bins = _targets
            else:
                # --- Set scalar to bins if dataset is the regression task ---
                bins = 20
            
            # --- Create histograms ---
            if (self.train_y is not None):
                self.target_distributions['train'] = {}
                hist_y, hist_x = np.histogram(self.train_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['train']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['train']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['train'] = None
            
            if (self.validation_y is not None):
                self.target_distributions['validation'] = {}
                hist_y, hist_x = np.histogram(self.validation_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['validation']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['validation']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['validation'] = None
            
            if (self.test_y is not None):
                self.target_distributions['test'] = {}
                hist_y, hist_x = np.histogram(self.test_y, bins=bins)
                if (self.dataset_type in ['img_reg', 'table_reg']):
                    hist_x = np.round(hist_x, decimals=2)
                self.target_distributions['test']['hist_y'] = (hist_y / np.sum(hist_y)).tolist()
                self.target_distributions['test']['hist_x'] = hist_x.tolist()[:-1]
            else:
                self.target_distributions['test'] = None
            
#---------------------------------
# クラス; CIFAR-10データセット取得
#---------------------------------
class DataLoaderCIFAR10(DataLoader):
    """DataLoaderCIFAR10
    
    CIFAR-10データセットのロード用クラス
    """
    
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
        self.verified = True
        
        # --- download dataset and extract ---
        if (download):
            logging.debug('[DataLoaderCIFAR10] {}'.format(dataset_dir))
            os.makedirs(dataset_dir, exist_ok=True)
            if (not Path(dataset_dir, 'cifar-10-batches-py').exists()):
                url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                save_file = self.file_download(dataset_dir, url)
                
                with tarfile.open(save_file) as tar:
                    # --- CVE-2007-4559 start ---
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = Path(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    safe_extract(tar, path=dataset_dir)
                    # --- CVE-2007-4559 end ---
            else:
                logging.debug('CIFAR-10 dataset is exists (Skip Download)')
        dataset_dir = Path(dataset_dir, 'cifar-10-batches-py')
            
        # --- load training data ---
        train_data_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        dict_data = unpickle(Path(dataset_dir, train_data_list[0]))
        train_x = dict_data[b'data']
        train_y = dict_data[b'labels'].copy()
        for train_data in train_data_list[1:]:
            dict_data = unpickle(Path(dataset_dir, train_data))
            train_x = np.vstack((train_x, dict_data[b'data']))
            train_y = np.hstack((train_y, dict_data[b'labels']))
        
        # --- load test data ---
        test_data = "test_batch"
        dict_data = unpickle(Path(dataset_dir, test_data))
        test_x = dict_data[b'data']
        test_y = dict_data[b'labels'].copy()
        
        # --- transpose: [N, C, H, W] -> [N, H, W, C] ---
        self.train_x = train_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.test_x = test_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # --- labels ---
        if (self.one_hot):
            identity = np.eye(10, dtype=np.int)
            self.train_y = np.array([identity[i] for i in train_y])
            self.test_y = np.array([identity[i] for i in test_y])
        else:
            self.train_y = np.array(train_y)
            self.test_y = np.array(test_y)
        
        # --- 学習データとバリデーションデータを分割 ---
        self.split_train_val(validation_split)
        
        # --- 出力次元数を保持 ---
        self.output_dims = 10
        
        # --- 画像分類タスク ---
        self.dataset_type = 'img_clf'
        
        return
        
#---------------------------------
# クラス; MNISTデータセット取得
#---------------------------------
class DataLoaderMNIST(DataLoader):
    """DataLoaderMNIST
    
    MNISTデータセットのロード用クラス
    """
    
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
        self.verified = True
        
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
                if (not Path(dataset_dir, mnist_file).exists()):
                    url = 'http://yann.lecun.com/exdb/mnist/' + mnist_file
                    save_file = self.file_download(dataset_dir, url)
                    
                    with gzip.open(save_file, 'rb') as gz:
                        gz_content = gz.read()
                    
                    save_file = Path(dataset_dir, mnist_file[:-3])
                    with open(save_file, 'wb') as f:
                        f.write(gz_content)
                    
                else:
                    logging.debug('{} is exists (Skip Download)'.format(mnist_file))
            
        # --- load training data ---
        f = open(Path(dataset_dir, 'train-images-idx3-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
        img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
        
        if (flatten):
            self.train_x = byte_data[16:].reshape(n_items, -1)
        else:
            self.train_x = byte_data[16:].reshape(n_items, img_h, img_w, 1)
        
        # --- load training label ---
        f = open(Path(dataset_dir, 'train-labels-idx1-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        
        self.train_y = byte_data[8:]
        if (self.one_hot):
            identity = np.eye(10, dtype=np.int)
            self.train_y = np.array([identity[i] for i in self.train_y])
        
        # --- load test data ---
        f = open(Path(dataset_dir, 't10k-images-idx3-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        img_h = (byte_data[8] << 24) | (byte_data[9] << 16) | (byte_data[10] << 8) | (byte_data[11])
        img_w = (byte_data[12] << 24) | (byte_data[13] << 16) | (byte_data[14] << 8) | (byte_data[15])
        
        if (flatten):
            self.test_x = byte_data[16:].reshape(n_items, -1)
        else:
            self.test_x = byte_data[16:].reshape(n_items, img_h, img_w, 1)
        
        # --- load test label ---
        f = open(Path(dataset_dir, 't10k-labels-idx1-ubyte'))
        byte_data = np.fromfile(f, dtype=np.uint8)
        
        n_items = (byte_data[4] << 24) | (byte_data[5] << 16) | (byte_data[6] << 8) | (byte_data[7])
        
        self.test_y = byte_data[8:]
        if (self.one_hot):
            self.test_y = np.array([identity[i] for i in self.test_y])
        
        # --- 学習データとバリデーションデータを分割 ---
        self.split_train_val(validation_split)
        
        # --- 出力次元数を保持 ---
        self.output_dims = 10
        
        # --- 画像分類タスク ---
        self.dataset_type = 'img_clf'
        
        return
    

#---------------------------------
# クラス; California Housingデータセット取得
#---------------------------------
class DataLoaderCaliforniaHousing(DataLoader):
    """DataLoaderCaliforniaHousing
    
    カリフォルニア住宅価格予測用データセットのロード用クラス
    """
    
    # --- コンストラクタ ---
    def __init__(self, dataset_dir, validation_size=0.2, test_size=0.3):
        from sklearn.datasets import fetch_california_housing
        
        # --- initialize super class ---
        super().__init__()
        
        self.one_hot = False
        self.output_dims = 1
        self.verified = True
        self.dataset_type = 'table_reg'
        
        california_housing = fetch_california_housing(data_home=dataset_dir)
        self.train_x = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        self.train_y = pd.DataFrame(california_housing.target, columns=['TARGET'])
        
        self.train_valid_test_split(validation_size=validation_size, test_size=test_size)
        
        return

#---------------------------------
# クラス; カスタムデータセット取得
#---------------------------------
class DataLoaderCustom(DataLoader):
    """DataLoaderCustom
    
    Data loader for custom dataset(user dataset).
    
    Attributes:
        train_x (numpy.ndarray): Train images
        train_y (numpy.ndarray): Train labels
        validation_x (numpy.ndarray): Validation images
        validation_y (numpy.ndarray): Validation labels
        test_x (numpy.ndarray): Test images
        test_y (numpy.ndarray): Test labels
        one_hot (bool): one hot or not about labels style
        output_dims (int): output layer dimensions
        verified (bool): verified status (True: OK, False: NG)
    
    """
    
    def __init__(self):
        """constructor"""
        
        # --- initialize super class ---
        super().__init__()
        
        self.train_x = None
        self.train_y = None
        self.validation_x = None
        self.validation_y = None
        self.test_x = None
        self.test_y = None
        self.one_hot = True
        self.output_dims = 1
        
        return
    
    def load_data(self, meta_dir, train_dir, validation_dir=None, test_dir=None, validation_split=0.0, flatten=False, one_hot=False):
        """load_data
        
        データをロードしてクラス変数へ設定する
        
        Args:
            train_dir (PosixPath): 学習データセットのディレクトリ
            validation_dir (PosixPath): バリデーションデータセットのディレクトリ
            test_dir (PosixPath): テストデータセットのディレクトリ
            validation_split (float): validation dataとして使用する学習データの比率(0.0 ～ 1.0)．validation_dirが指定されている場合は，validation_splitは無視する
            flatten (bool): 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
            one_hot (bool): one hot形式(=True)かラベルインデックス(=False;default)かを選択する
        
        Returns:
            None
        """
        
        def _load_image_data(data_dir, key_name='img_file'):
            """_load_image_data
                カスタムデータセットを読み込み，画像とラベルを返す
                
                [引数説明]
                  * data_dir: カスタムデータセットのディレクトリ
            """
            
            # --- Load json data ---
            json_file = Path(data_dir, 'info.json')
            df_data = pd.read_json(json_file, orient='records')
            
            # --- Get channel ---
            img = Image.open(Path(data_dir, df_data[key_name][0]))
            if (len(img.size) == 2):
                # --- Grayscale ---
                img_channel = 1
            else:
                # --- RGB ---
                img_channle = 3
            
            # --- Load images ---
            #   * Loading images should be use Pillow, because OpenCV cannot load the grayscale image as grayscale
            #     ex) Image.open('grayscale.png').size = [H, W]
            #         cv2.imread('grayscale.png').shape = [H, W, C] (Auto convert from grayscale to color)
            images = []
            labels = []
            if (img_channel == 1):
                for index, data_ in df_data.iterrows():
                    images.append(np.array(Image.open(Path(data_dir, data_[key_name])))[:, :, np.newaxis].tolist())
                    labels.append(data_['target'])
            else:
                for index, data_ in df_data.iterrows():
                    images.append(np.array(Image.open(Path(data_dir, data_[key_name]))).tolist())
                    labels.append(data_['target'])
            
            return np.array(images), np.array(labels)
        
        def _load_table_data(data_dir, feature_names):
            """ _load_table_data
            
            This function loads table data from ``info.json``
            
            Args:
                data_dir (string): data directory ``info.json`` in.
                feature_names (list of string): feature names
            
            """
            # --- Load json data ---
            json_file = Path(data_dir, 'info.json')
            df_data = pd.read_json(json_file, orient='records')
            
            # --- Split input feature and target
            X_data = df_data[feature_names]
            if ("target" in df_data.columns):
                y_data = df_data["target"]
            else:
                y_data = None
            
            return X_data, y_data
            
        # --- set parameters from arguments ---
        self.one_hot = one_hot
        
        # --- load meta data ---
        df_meta = pd.read_json(Path(meta_dir, 'info.json'), typ='series')
        if ((df_meta['task'] == 'classification') and (df_meta['input_type'] == 'image_data')):
            self.dataset_type = 'img_clf'
        elif ((df_meta['task'] == 'regression') and (df_meta['input_type'] == 'image_data')):
            self.dataset_type = 'img_reg'
        elif ((df_meta['task'] == 'classification') and (df_meta['input_type'] == 'table_data')):
            self.dataset_type = 'table_clf'
        elif ((df_meta['task'] == 'regression') and (df_meta['input_type'] == 'table_data')):
            self.dataset_type = 'table_reg'
        else:
            self.dataset_type = None
        
        if (df_meta['input_type'] == 'image_data'):
            for key in df_meta['keys']:
                if (key['type'] == 'image_file'):
                    key_name = key['name']
                    break
            
            # --- load training data ---
            self.train_x, self.train_y = _load_image_data(train_dir, key_name=key_name)
            
            # --- load validation data ---
            if (validation_dir is not None):
                self.validation_x, self.validation_y = _load_image_data(validation_dir, key_name=key_name)
            else:
                self.split_train_val(validation_split)
                
            # --- load test data ---
            if (test_dir is not None):
                self.test_x, self.test_y = _load_image_data(test_dir, key_name=key_name)
            
            # --- set output dims ---
            self.output_dims = len(np.unique(self.train_y))
            
        elif (df_meta['input_type'] == 'table_data'):
            # --- load input feature names ---
            feature_names = []
            for key in df_meta['keys']:
                feature_names.append(key['name'])
            
            # --- load training data ---
            self.train_x, self.train_y = _load_table_data(train_dir, feature_names)
            
            # --- load validation data ---
            if (validation_dir is not None):
                self.validation_x, self.validation_y = _load_table_data(validation_dir, feature_names)
            
            # --- load test data ---
            if (test_dir is not None):
                self.test_x, self.test_y = _load_table_data(test_dir, feature_names)
            
        return
    
    def verify(self, meta_dir, train_dir, validation_dir=None, test_dir=None):
        """verify
        
        データ形式の整合検証
        
        Args:
            meta_dir (PosixPath): Meta data (zip extracted)
            train_dir (PosixPath): Train data (zip extracted)
            validation_dir (PosixPath): Validation data (zip extracted)
            test_dir (PosixPath): Test data (zip extracted)
        
        Returns:
            bool: Result of verification (True: OK, False: NG)
        
        """
        
        self.verified = False
        
        if (not Path(meta_dir, 'info.json').exists()):
            return False
            
        if (not Path(train_dir, 'info.json').exists()):
            return False
        
        if (validation_dir is not None):
            if (not Path(validation_dir, 'info.json').exists()):
                return False
                
        if (test_dir is not None):
            if (not Path(test_dir, 'info.json').exists()):
                return False
        
        self.verified = True
        return True

