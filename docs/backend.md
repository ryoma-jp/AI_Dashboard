# バックエンド (Django/DRF)

## 構成

- Django 設定: `django_project/project/settings.py`
- ルーティング: `django_project/project/urls.py`, `django_project/app/urls.py`, `django_project/app/urls_api.py`
- モデル: `django_project/app/models.py`
- API: `django_project/app/apis.py`

## ルーティング方針

`project/urls.py` はアプリ URL を `app.urls` に集約し、`app/urls.py` が画面遷移を担当します。
API は `app/urls_api.py` で定義し、主に以下を提供します。

- `api/get_project_list`
- `api/get_dataset_list`
- `api/jobs/<id>/`

## 主要ドメインモデル

- `Project`: プロジェクト情報
- `Dataset`: データセットとアップロード ZIP 管理
- `AIModelSDK`: カスタム SDK 管理
- `MlModel`: 学習対象モデルと実行状態
- `OperationJob`, `OperationStep`: 長時間処理の進捗管理

## 処理特性

- `Dataset` と `AIModelSDK` は `save()` 内で一度ダミー保存を行い、ID 確定後に保存先パスを構築
- `OperationJob` / `OperationStep` によって、学習や推論の進捗を UI/API で追跡
- `apis.py` のジョブ詳細取得 API では FIFO 完了通知の取り込みを実施

## 設定のポイント

`settings.py` は環境変数ベースで構成され、次のディレクトリ設定を使用します。

- `DATASET_DIR`
- `MODEL_DIR`
- `CONFIG_DIR`
- `AI_MODEL_SDK_DIR`

これらは `media` 配下の実データ構成と連動します。
