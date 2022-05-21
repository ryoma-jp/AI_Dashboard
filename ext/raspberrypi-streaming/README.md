# Raspberry Piでカメラ映像をストリーミング配信する

## 主な用途

* AI Dashboard上にカメラ映像を表示可能にする
* Raspberry Pi上で推定した結果を重畳した映像をストリーミング配信できるようにする

## ストリーミング配信の方法

mjpg-streamerを使う

## 環境構築手順

1. prepare
    ```
    $ sudo apt update
    $ sudo apt install -y cmake g++ wget unzip libv4l-dev libjpeg-dev imagemagick
    ```
1. install OpenCV
    ```
    $ wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/3.4.17.zip
    $ unzip opencv.zip
    $ mkdir -p build && cd build
    $ cmake ../opencv-3.4.17
    $ cmake --build .
    $ sudo make install
    ```
1. install mjpg-streamer  
    ```
    $ git clone https://github.com/jacksonliam/mjpg-streamer.git
    $ cd mjpg-streamer
    $ git checkout 310b29f4a94c46652b20c4b7b6e5cf24e532af39
    $ cd mjpg-streamer-experimental
    $ ln -s plugins/input_opencv/cmake-modules <path/to/OpenCV/build>
    $ (edit CMakeLists.txt below)
        diff --git a/mjpg-streamer-experimental/plugins/input_opencv/CMakeLists.txt b/mjpg-streamer-experimental/plugins/input_opencv/CMakeLists.txt
        index 9adae85..b852003 100644
        --- a/mjpg-streamer-experimental/plugins/input_opencv/CMakeLists.txt
        +++ b/mjpg-streamer-experimental/plugins/input_opencv/CMakeLists.txt
        @@ -1,10 +1,10 @@
        
         # TODO: which components do I need?
         # To fix the error: "undefined symbol: _ZN2cv12VideoCaptureC1Ev"
        +set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
         find_package(OpenCV COMPONENTS core imgproc highgui videoio)
        
        -MJPG_STREAMER_PLUGIN_OPTION(input_opencv "OpenCV input plugin"
        -                            ONLYIF OpenCV_FOUND ${OpenCV_VERSION_MAJOR} EQUAL 3)
        +MJPG_STREAMER_PLUGIN_OPTION(input_opencv "OpenCV input plugin")
        
         if (PLUGIN_INPUT_OPENCV)
             enable_language(CXX)
        diff --git a/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/CMakeLists.txt b/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/CMakeLists.txt
        index 6782dcb..9a5c713 100644
        --- a/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/CMakeLists.txt
        +++ b/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_py/CMakeLists.txt
        @@ -8,8 +8,7 @@ set(Python_ADDITIONAL_VERSIONS 3.4 3.5)
         find_package(PythonLibs)
         find_package(Numpy)
        
        -MJPG_STREAMER_PLUGIN_OPTION(cvfilter_py "OpenCV python filter"
        -                            ONLYIF PYTHONLIBS_FOUND NUMPY_FOUND ${PYTHON_VERSION_MAJOR} EQUAL 3)
        +MJPG_STREAMER_PLUGIN_OPTION(cvfilter_py "OpenCV python filter")
        
         if (PLUGIN_CVFILTER_PY)
             include_directories(${PYTHON_INCLUDE_DIRS})
    $ make; sudo make install
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
1. start server (OpenCV)
    ```
    $ ./mjpg_streamer -i "./input_opencv.so --filter ./cvfilter_py.so --fargs ./plugins/input_opencv/filters/cvfilter_py/example_filter.py" -o "./output_http.so -w ./www"
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
* [ラズパイでmjpg-streamerのopencvプラグインを使う＋OpenCVのビルドインストール(備忘録)](https://qiita.com/zuttonetetai/items/e0c4b13a6012b285db01)
* [Raspberry Pi + Python 3 に OpenCV 3 をなるべく簡単にインストールする](https://qiita.com/masaru/items/658b24b0806144cfeb1c)
** ```sudo pip3 install opencv-python==3.4.17.63```のように，opencv-pythonのバージョンを指定する
** デフォルトのままでは，OpenCV 4系がインストールされてしまう(2022.5.19現在)

