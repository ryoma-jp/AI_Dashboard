#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import os
import argparse
import json
from ml_params import MlParams, MlParams_MNIST, MlParams_CIFAR10, MlParams_CaliforniaHousing

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数: ArgParser
#   * 引数処理関数
#---------------------------------
def ArgParser():
    parser = argparse.ArgumentParser(description='機械学習用設定ファイル(json)を生成するツール',
                formatter_class=argparse.RawTextHelpFormatter)

    # --- 引数を追加 ---
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='output', required=False, \
            help='jsonファイルの出力ディレクトリ')

    args = parser.parse_args()

    return args

#---------------------------------
# 関数: main
#   * メイン関数
#---------------------------------
def main():
    # --- 引数処理 ---
    args = ArgParser()
    print(f'args.output_dir : {args.output_dir}')

    # --- 出力ディレクトリ生成 ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- デフォルトパラメータ生成(Blank) ---
    blank_params = MlParams()
    with open(os.path.join(args.output_dir, 'config_blank.json'), 'w') as f:
        json.dump(blank_params.params, f, ensure_ascii=False, indent=4)
    
    # --- デフォルトパラメータ生成(MNIST) ---
    mnist_params = MlParams_MNIST()
    with open(os.path.join(args.output_dir, 'config_mnist.json'), 'w') as f:
        json.dump(mnist_params.params, f, ensure_ascii=False, indent=4)
    
    # --- デフォルトパラメータ生成(CIFAR-10) ---
    cifar10_params = MlParams_CIFAR10()
    with open(os.path.join(args.output_dir, 'config_cifar10.json'), 'w') as f:
        json.dump(cifar10_params.params, f, ensure_ascii=False, indent=4)

    # --- デフォルトパラメータ生成(California Housing) ---
    california_housing_params = MlParams_CaliforniaHousing()
    with open(os.path.join(args.output_dir, 'config_california_housing.json'), 'w') as f:
        json.dump(california_housing_params.params, f, ensure_ascii=False, indent=4)

    return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
    main()

