# フロントエンド (Django Template/Static)

## 構成

- テンプレート: `django_project/templates/`
- 静的ファイル: `django_project/static/css/`, `django_project/static/js/`

## 主要画面

- ダッシュボード: `index.html`
- データセット一覧/詳細: `dataset.html`, `dataset_detail.html`
- 学習画面: `training.html`
- 推論画面: `inference.html`
- モデル/プロジェクト編集: `model_*.html`, `project_*.html`
- ストリーミング画面: `view_streaming.html`

## View との対応

画面は `django_project/app/views_*.py` からレンダリングされます。主な責務は次の通りです。

- `views_project.py`: プロジェクトとモデルの CRUD
- `views_dataset.py`: データセット操作と詳細表示
- `views_training.py`: 学習ジョブの起動/停止、学習パラメータ設定
- `views_inference.py`: 推論処理と出力ダウンロード
- `views_view_streaming.py`: USB カメラ/YouTube の表示

## 改修時の注意

- URL 追加時は `app/urls.py` と対応 View、Template を同時更新する
- 長時間処理の進捗表示が必要な UI は `OperationJob` API と整合させる
- 既存テンプレート構造を維持し、不要な全体レイアウト変更は避ける
