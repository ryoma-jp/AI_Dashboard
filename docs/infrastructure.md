# インフラ/実行環境

## コンテナ構成

`docker-compose.yml` の主要サービス:

- `web`: Django + Gunicorn (`project.wsgi:application`)
- `db`: PostgreSQL 13
- `nginx`: 80 番ポート公開 (ホスト 1317)
- `jupyter_notebook_tf`: TensorFlow Notebook

## ボリューム/バインド

- 永続データ: `postgres_data`, `static_volume`, `env_data`
- バインドマウント:
  - `django_project/media/` -> `/home/app/media`
  - `tools/`, `test/`
  - デバッグ用途で `app/`, `project/`, `templates/`

## nginx 設定

`nginx/nginx.conf` では:

- `web:8000` へのリバースプロキシ
- `/static/` を `static_volume` から配信
- 学習/推論の長時間処理を考慮した timeout 設定

## 起動の基本手順

1. `.env.dev` を準備
2. `docker-compose build`
3. `docker-compose up -d`
4. `docker-compose exec web python manage.py migrate --no-input`
5. `docker-compose exec web python manage.py collectstatic --no-input`

## 運用メモ

- ローカル検証は Docker 実行を基本とする
- GPU 利用が必要なサービスは `deploy.resources.reservations.devices` 設定を参照する
- DB 初期化は `docker compose down --volumes` を利用する
