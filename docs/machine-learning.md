# 機械学習パイプライン

## 実装配置

- エントリポイント: `django_project/app/machine_learning/main.py`
- 学習/推論補助: `django_project/app/machine_learning/ml_train_main.py`, `ml_inference_main.py`
- 共通ライブラリ: `django_project/app/machine_learning/lib/`

## `main.py` の概要

`main.py` は `--mode` と `--config` を受け取り、次を実行します。

1. JSON 設定の読み込み
2. データセット (`dataset.pkl`) のロードと前処理
3. モデル種別ごとの Trainer 構築
4. 学習または予測の実行
5. 必要に応じた精度計算や特徴量取得

## 対応モデル (抜粋)

- MLP
- SimpleCNN / DeepCNN
- SimpleResNet / DeepResNet
- YOLOv3 / YOLOv3_Tiny
- LightGBM

## 設定連携

学習条件や入出力先は `config.json` 由来の値を参照し、Web 側が生成する設定と連動します。
FIFO パスを介して Web 側制御と連携するため、長時間処理ではジョブ状態管理とあわせて設計します。

## 補助リソース

- 設定生成スクリプト: `tools/create_config/`
- ML 検証サンプル: `test/`
- Notebook 用ライブラリ共有: `django_project/app/machine_learning/lib/`
