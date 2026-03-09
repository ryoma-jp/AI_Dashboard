# AI Dashboard Docs

AI Dashboard の設計情報と実装ガイドを技術別にまとめたドキュメントです。

## 目次

- [アーキテクチャ概要](./architecture.md)
- [バックエンド (Django/DRF)](./backend.md)
- [フロントエンド (Django Template/Static)](./frontend.md)
- [機械学習パイプライン](./machine-learning.md)
- [インフラ/実行環境](./infrastructure.md)
- [運用メモ](./operations.md)
- [外部参照リンク集](./reference-links.md)

## 読み進め方

1. まず [アーキテクチャ概要](./architecture.md) で全体像を把握する
2. Web アプリ改修時は [バックエンド](./backend.md) と [フロントエンド](./frontend.md) を参照する
3. 学習/推論ロジック変更時は [機械学習パイプライン](./machine-learning.md) を参照する
4. 起動構成やデプロイ関連は [インフラ/実行環境](./infrastructure.md) を参照する
5. 運用時の補足は [運用メモ](./operations.md) を参照する
6. 実装時の追加調査は [外部参照リンク集](./reference-links.md) を補助的に参照する

## 重複整理ポリシー

- プロジェクト固有の設計/実装方針は `architecture.md` から `infrastructure.md` までの技術文書に集約
- 一般的な運用ノウハウは `operations.md` に集約
- 外部記事や公式サイトへのリンクは `reference-links.md` に集約

## 関連ファイル

- ルート README: [../README.md](../README.md)
- タスク管理: [../.github/tasks/current.md](../.github/tasks/current.md)
