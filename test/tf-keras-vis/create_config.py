import os
import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from tensorflow.keras.applications import VGG16
from machine_learning.lib.trainer.tf_models.yolov3 import models as yolov3_models

def ArgParser():
    parser = argparse.ArgumentParser(description='Create config file for tf-keras-vis application',
                formatter_class=argparse.RawTextHelpFormatter)

    model_list = [
        'VGG16',
        'MobileNetv2',
        'CenterNetHourGlass104',
        'YOLOv3',
    ]
    parser.add_argument('--model', dest='model', type=str, choices=model_list, default='VGG16', \
            help='Model\n'
                 '  - MobileNetv2 and CenterNetHourGlass104 does not work')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='config', \
            help='Output directory')

    args = parser.parse_args()

    return args

def save_config(layer, model, layer_index, layer_name, output_dir):
    """Save Config
    
    Save config file
    
    Args:
        layer: Layer object
        model: Model name
        layer_index: Layer index
        layer_name: Layer name
        output_dir: Output directory path to save config file
    
    """
    if (layer.__class__.__name__ == 'Conv2D'):
        config = {
            'model': model,
            'index': layer_index,
            'name': layer_name,
        }
        
        config_path = Path(output_dir, f'layer{layer_index}_{layer_name}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)


def main():
    args = ArgParser()
    print(f'args.model: {args.model}')
    print(f'args.output_dir: {args.output_dir}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if (args.model == 'VGG16'):
        model = VGG16()
    elif (args.model == 'MobileNetv2'):
        model_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
        image_shape = (224, 224)
        
        model = tf.keras.Sequential([
            hub.KerasLayer(model_url, input_shape=image_shape+(3,), trainable=False)
        ])
    elif (args.model == 'CenterNetHourGlass104'):
        model_url = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'
        image_shape = (512, 512)
        dtype = tf.uint8
        
        model = tf.keras.Sequential([
            hub.KerasLayer(model_url, input_shape=image_shape+(3,), trainable=False, dtype=dtype)
        ])
    elif (args.model == 'YOLOv3'):
        model = yolov3_models.YoloV3(size=416, classes=80, training=True)
    else:
        assert f'{args.model} is not supported'
    
    model.summary()
    
    for i, layer in enumerate(model.layers):
        layer_config = layer.get_config()
        #print(f'layer{i} config: {layer_config}')
        if ('layers' in layer_config.keys()):
            for j, _layer in enumerate(layer.layers):
                save_config(_layer, args.model, f'{i}-{j}', _layer.name, args.output_dir)
        else:
            save_config(layer, args.model, i, layer.name, args.output_dir)
            
if __name__=='__main__':
    main()
