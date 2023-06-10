# Sample program of tf-keras-viz

## Usage

### create_config.py

```
usage: create_config.py [-h] [--model {VGG16}] [--output_dir OUTPUT_DIR]

Create config file for tf-keras-vis application

optional arguments:
  -h, --help            show this help message and exit
  --model {VGG16}       Model
  --output_dir OUTPUT_DIR
                        Output directory
```

### tf-keras-vis-app.py

```
usage: tf-keras-vis-app.py [-h] --config CONFIG

Sample application for tf-keras-vis

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Config file path(*.json)
                     {
                       "model": <model name(VGG16)>
                       "index": <layer index(0, 1, ...)>
                       "name": <layer name>
                     }
```

## Execute sample

### VGG16

```
$ run_vgg16.sh
```

