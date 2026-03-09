# アーキテクチャ概要

## 目的

AI Dashboard は、機械学習のデータ準備・学習・推論を Web UI から実行し、関連成果物をプロジェクト単位で管理するための Django ベースアプリケーションです。

## 技術スタック

- Web アプリ: Django 3.x
- API: Django REST Framework
- DB: PostgreSQL
- App サーバ: Gunicorn
- リバースプロキシ: nginx
- 実験環境: Jupyter Notebook (TensorFlow)
- ML 実装: `django_project/app/machine_learning/`

## ディレクトリ責務

- `django_project/app/`: ドメインモデル、画面 View、API、ML 呼び出し起点
- `django_project/project/`: Django 設定、URL ルーティング
- `django_project/templates/`: 画面テンプレート
- `django_project/static/`: 静的ファイル
- `django_project/media/`: データセット、モデル、設定、Notebook 実データ
- `nginx/`: nginx 設定
- `tools/`: 設定生成など補助スクリプト
- `test/`: 検証用スクリプト

## 実行構成 (Docker Compose)

`docker-compose.yml` では次の主要サービスを定義します。

- `web`: Django + Gunicorn
- `db`: PostgreSQL
- `nginx`: フロントエンド公開
- `jupyter_notebook_tf`: Notebook 実行環境

詳細は [インフラ/実行環境](./infrastructure.md) を参照してください。

## 主要機能モジュール

- プロジェクト/モデル管理: `views_project.py`
- データセット管理: `views_dataset.py`
- 学習実行: `views_training.py`
- 推論実行: `views_inference.py`
- ストリーミング表示: `views_view_streaming.py`
- API: `apis.py`, `urls_api.py`

## 参考図

- ER 図: `../figure/erd_ml_project_database.png`
- ディレクトリ構成図: `../figure/directory_structure.png`
