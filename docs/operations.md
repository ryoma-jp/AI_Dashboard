# 運用メモ

README から分離した運用上の補足情報をまとめます。
設計や実装の主情報は以下を参照してください。

- [アーキテクチャ概要](./architecture.md)
- [インフラ/実行環境](./infrastructure.md)
- [バックエンド (Django/DRF)](./backend.md)
- [機械学習パイプライン](./machine-learning.md)

## 環境構築補足

環境構築の基本手順は README のクイックスタートを参照し、本ドキュメントは運用時の補足に限定します。

## docker-compose で GPU を有効にする

- [Enabling GPU access with Compose](https://docs.docker.com/compose/gpu-support/)
- [docker-composeでgpuを使う方法](https://qiita.com/hgoj/items/1ba050c7d73e0dd5ef90)

## docker-compose の更新例

```bash
which docker-compose
sudo apt remove docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-Linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

- [Docker: docker-compose を最新版にアップグレードする](https://qiita.com/kottyan/items/c892b525b14f293ab7b3)

## Docker ビルドコンテキストの調整

### 問題

Dockerfile はビルド起点より上位のディレクトリを直接参照できないため、コピーやマウントで参照エラーが起きる場合があります。

### 対策

`docker-compose.yml` で `build.context` と `build.dockerfile` を明示し、必要な最上位ディレクトリをビルド起点にします。

```yaml
build:
  context: <build-context-path>
  dockerfile: <relative-path-from-context>
```

- [Docker Compose で Dockerfile ビルド時に親ディレクトリを扱う](https://zukucode.com/2020/08/docker-compose-parent-directory.html)

## docker-compose への環境変数注入

ホスト IP やホスト名など環境依存値を動的に使いたい場合は、`docker-compose.yml` のみで完結させず、環境変数の読み込み順序を設計します。

- [docker-compose で環境変数を設定する](https://qiita.com/reflet/items/a96752b4cd3838ea6932)
- [Docker Compose v2 でインラインコマンドが使えないケース](https://ema-hiro.hatenablog.com/entry/2021/10/11/131455)

## Jupyter Notebook token 固定メモ

- ブラウザ起動時の token 入力を固定したい場合は `.jupyter/jupyter_notebook_config.py` で token を設定します。
- [Jupyter notebook のパスワード設定](https://qiita.com/SaitoTsutomu/items/aee41edf1a990cad5be6)

## PlantUML 表示メモ

GitHub の Markdown では PlantUML をそのまま図表示できないため、運用方針を決めておくと差分管理しやすくなります。

- [PlantUML Server](https://plantuml.com/ja/server)
- [GitHub Markdown で PlantUML を表示する Chrome 拡張](https://dev.classmethod.jp/articles/chrome-extension-plantuml-in-github-markdown/)
- [GitHub README にシーケンス図を埋め込む](https://qiita.com/takke/items/86a5ddf145cf9693b6e9)
- [PlantUML and GitLab](https://docs.gitlab.com/ee/administration/integration/plantuml.html)

## YouTube 動画入力メモ

YouTube 動画を Python で取り込む用途では、以下の実績メモを残します。

- [cap_from_youtube](https://github.com/ibaiGorordo/cap_from_youtube) を採用
- 比較候補: [youtube-dl](https://github.com/ytdl-org/youtube-dl), [yt-dlp](https://github.com/yt-dlp/yt-dlp)
