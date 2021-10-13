#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import argparse
from django.core.management.utils import get_random_secret_key

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	parser = argparse.ArgumentParser(description='local_settings.pyを生成するツール',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--output_file', dest='output_file', type=str, default='local_settings.py', required=False, \
			help='local_settgins.pyの出力ファイル')

	args = parser.parse_args()

	return args

def main():
	# --- 引数処理 ---
	args = ArgParser()
	print('args.output_file : {}'.format(args.output_file))

	# --- local_settings.py生成 ---
	with open(args.output_file, 'w') as f:
		f.write('SECRET_KEY = \'{}\''.format(get_random_secret_key()))

	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

