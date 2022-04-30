# Raspberry Piでカメラ映像をストリーミング配信する

## 主な用途

* AI Dashboard上にカメラ映像を表示可能にする
* Raspberry Pi上で推定した結果を重畳した映像をストリーミング配信できるようにする

## ストリーミング配信の方法

mjpg-streamerを使う

## 環境構築手順

1. install mjpg-streamer  
  ```
  $ sudo apt update
  $ sudo apt install -y cmake libv4l-dev libjpeg-dev imagemagick
  $ git clone https://github.com/jacksonliam/mjpg-streamer.git
  $ cd mjpg-streamer
  $ git checkout a9340754074554553bacf3e865819baaa52f28ce
  $ cd mjpg-streamer-experimental
  $ sudo make; sudo make install
  ```
1. edit script  
  下記の行が有効となるように編集
  ```
  ./mjpg_streamer -i "./input_uvc.so -n -f 30 -r 640x480 -d /dev/video0"  -o "./output_http.so -w ./www"
  ```
1. start server  
  ```
  $ ./start.sh
  ```

## ストリーミング配信の確認

ブラウザで，```	http://(Raspberry PiのIPアドレス):(PORT番号)```にアクセスする  
デフォルトのPORT番号は```8080```

## ストリーミング配信映像のHTML埋め込み方法

```
<img src="http://(Raspberry PiのIPアドレス):(PORT番号)/?action=stream">
```

## 参照

* [Raspberry PiとWebカメラでストリーミング配信してみた](https://www.ecomottblog.com/?p=8791)
