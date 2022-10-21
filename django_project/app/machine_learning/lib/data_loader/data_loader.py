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

from pathlib import Path

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
    
    """
    # --- コンストラクタ ---
    def __init__(self):
        self.one_hot = True
        self.output_dims = -1
        self.verified = False
        self.dataset_type = 'img_clf'
        
        return
    
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
    
    # --- 標準化・正規化 ---
    #  * mode: 正規化手法
    #     max: データの絶対値の最大が1.0となるように正規化(最大値で除算)
    #     max-min: データの最大が1.0最小が0.0となるように正規化(最大値と最小値を用いて算出)
    #     z-score: 標準化(平均と標準偏差を用いて算出)
    #
    #  [Reference]
    #  * 標準化と正規化: https://aiacademy.jp/texts/show/?id=555
    def normalization(self, mode):
        """normalization
        
        データの正規化を実行する
        
        Args:
            mode (str): 正規化方法を指定する
                - 'max': データの絶対値の最大が1.0となるように正規化(最大値で除算)
                - 'max-min': データの最大が1.0最小が0.0となるように正規化(最大値と最小値を用いて算出)
                - 'z-score': 標準化(平均と標準偏差を用いて算出)
        
        """
        if (mode == 'max'):
            train_norm = self.train_x / 255.
            validation_norm = self.validation_x / 255.
            test_norm = self.test_x / 255.
        elif (mode == 'max-min'):
            train_min = np.min(self.train_x)
            train_diff = np.max(self.train_x) - np.min(self.train_x)
            
            train_norm = (self.train_x - train_min) / train_diff
            validation_norm = (self.validation_x - train_min) / train_diff
            test_norm = (self.test_x - train_min) / train_diff
        elif (mode == 'z-score'):
            train_mean = np.mean(self.train_x)
            train_std = np.std(self.train_x)
            
            train_norm = (self.train_x - train_mean) / train_std
            validation_norm = (self.validation_x - train_mean) / train_std
            test_norm = (self.test_x - train_mean) / train_std
        else:
            logging.debug('[ERROR] Unknown data normalization mode: {}'.format(mode))
            quit()
        
        return train_norm, validation_norm, test_norm
        
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
    
    def convert_label_encoding(self, one_hot=True):
        """Convert Label Encoding
        
        正解ラベルをOne Hot表現とインデックス表現を相互変換する
        
        Args:
            one_hot (bool): 変換先のインデックス表現を指定
        """
        
        if ((not self.one_hot) and (one_hot)):
            identity = np.eye(self.output_dims, dtype=np.int)
            self.train_y = np.array([identity[i] for i in self.train_y])
            self.validation_y = np.array([identity[i] for i in self.validation_y])
            self.test_y = np.array([identity[i] for i in self.test_y])
        elif ((self.one_hot) and (not one_hot)):
            self.train_y = self.train_y.argmax(axis=1)
            self.validation_y = self.validation_y.argmax(axis=1)
            self.test_y = self.test_y.argmax(axis=1)
            
        self.one_hot = one_hot
        
        return
    
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
    def __init__(self, dataset_dir, validation_split=0.2):
        from sklearn.datasets import fetch_california_housing
        
        self.one_hot = False
        self.output_dims = 1
        self.verified = False
        self.dataset_type = 'table_reg'
        
        california_housing = fetch_california_housing(data_home=dataset_dir)
        self.train_x = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
        self.train_y = pd.DataFrame(california_housing.target, columns=['TARGET'])
        
        self.split_train_val(validation_split)
        
        # T.B.D
        self.test_x = self.validation_x
        self.test_y = self.validation_y
        
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
        self.output_dims = 0
        self.verified = False
        
        return
    
    def load_data(self, train_dir, test_dir, validation_dir=None, validation_split=0.0, flatten=False, one_hot=False):
        """load_data
        
        データをロードしてクラス変数へ設定する
        
        Args:
            train_dir (PosixPath): 学習データセットのディレクトリ
            test_dir (PosixPath): テストデータセットのディレクトリ
            validation_dir (PosixPath): バリデーションデータセットのディレクトリ
            validation_split (float): validation dataとして使用する学習データの比率(0.0 ～ 1.0)．validation_dirが指定されている場合は，validation_splitは無視する
            flatten (bool): 入力形式を[N, H, W, C](=False;default)とするか[N, H*W*C](=True)とするかを選択する(T.B.D)
            one_hot (bool): one hot形式(=True)かラベルインデックス(=False;default)かを選択する
        
        Returns:
            None
        """
        
        def _load_data(data_dir):
            """_load_data
                カスタムデータセットを読み込み，画像とラベルを返す
                
                [引数説明]
                  * data_dir: カスタムデータセットのディレクトリ
            """
            json_file = Path(data_dir, 'info.json')
            df_data = pd.read_json(json_file, orient='records')
            
            img = cv2.imread(Path(data_dir, df_data['file'][0]))
            n_items = len(df_data)
            if (img.ndim) == 3:
                img_h, img_w, n_channel = img.shape
            
            images = []
            labels = []
            for data_ in df_data.itertuples():
                images.append(list(cv2.imread(Path(data_dir, data_.file))))
                labels.append(data_.class_id)
            
            return np.array(images), np.array(labels)
        
        # --- initialize super class ---
        super().__init__()
        self.one_hot = one_hot
        
        # --- load training data ---
        self.train_x, self.train_y = _load_data(train_dir)
        
        # --- load test data ---
        self.test_x, self.test_y = _load_data(test_dir)
        
        # --- load validation data ---
        if (validation_dir is not None):
            self.validation_x, self.validation_y = _load_data(validation_dir)
        else:
            self.split_train_val(validation_split)
            
        # --- 出力次元数を保持 ---
        self.output_dims = 10    # T.B.D
        
        return
    
    def verify(self, train_dir, validation_dir=None, test_dir=None):
        """verify
        
        データ形式の整合検証
        
        Args:
            train_dir (PosixPath): Train data (zip extracted)
            validation_dir (PosixPath): Validation data (zip extracted)
            test_dir (PosixPath): Test data (zip extracted)
        
        Returns:
            bool: Result of verification (True: OK, False: NG)
        
        """
        
        self.verified = False
        
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

