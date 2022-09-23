'''Create Local Settings

local_settings.pyを生成するツール
'''


#! -*- coding: utf-8 -*-

#---------------------------------
# モジュールのインポート
#---------------------------------
import argparse
import secrets

from django.core.management.utils import get_random_secret_key

#---------------------------------
# 定数定義
#---------------------------------

#---------------------------------
# 関数
#---------------------------------
def ArgParser():
	'''ArgParser
	
	引数解析
	'''
	parser = argparse.ArgumentParser(description='local_settings.pyを生成するツール',
				formatter_class=argparse.RawTextHelpFormatter)

	# --- 引数を追加 ---
	parser.add_argument('--output_file', dest='output_file', type=str, default='local_settings.py', required=False, \
			help='local_settgins.pyの出力ファイル')
	parser.add_argument('--jupyter_nb_config', dest='jupyter_nb_config', type=str, default=None, required=False, \
			help='Jupyter NotebookのDockerビルド用設定ファイルのパス')

	args = parser.parse_args()

	return args

def main():
	'''main
	
	メイン関数
	'''
	
	# --- 引数処理 ---
	args = ArgParser()
	print('args.output_file : {}'.format(args.output_file))
	print('args.jupyter_nb_config : {}'.format(args.jupyter_nb_config))

	# --- local_settings.py生成 ---
	with open(args.output_file, 'w') as f:
		f.write(f'SECRET_KEY = \'{get_random_secret_key()}\'\n')
		
		jupyter_nb_token = secrets.token_hex()
		f.write(f'JUPYTER_NB_TOKEN = \'{jupyter_nb_token}\'\n')
		
	# --- jupyter_notebook_config.py生成 ---
	if (args.jupyter_nb_config is not None):
		with open(args.jupyter_nb_config, 'w') as f:
			f.write(f'c.NotebookApp.token = \'{jupyter_nb_token}\'\n')
	
	return

#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	main()

