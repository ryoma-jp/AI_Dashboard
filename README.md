# AI Dashboard

## 概要

- Django ベースの機械学習ダッシュボードです。
- データセット管理、学習、推論を Web UI から実行できます。
- 詳細仕様、運用メモ、外部参照は docs/ 配下に集約しています。

## docs 参照

- [docs/README.md](./docs/README.md)
- [アーキテクチャ概要](./docs/architecture.md)
- [バックエンド (Django/DRF)](./docs/backend.md)
- [フロントエンド (Django Template/Static)](./docs/frontend.md)
- [機械学習パイプライン](./docs/machine-learning.md)
- [インフラ/実行環境](./docs/infrastructure.md)
- [運用メモ](./docs/operations.md)
- [外部参照リンク集](./docs/reference-links.md)

## クイックスタート

1. docker-compose build
2. docker-compose up -d
3. docker-compose exec web python manage.py migrate --no-input
4. docker-compose exec web python manage.py collectstatic --no-input
5. http://localhost:1317/ を開く
