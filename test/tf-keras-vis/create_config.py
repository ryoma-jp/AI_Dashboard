import os
import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from tensorflow.keras.applications import VGG16

def ArgParser():
    parser = argparse.ArgumentParser(description='Create config file for tf-keras-vis application',
                formatter_class=argparse.RawTextHelpFormatter)

    model_list = [
        'VGG16',
        'MobileNetv2',
        'CenterNetHourGlass104',
    ]
    parser.add_argument('--model', dest='model', type=str, choices=model_list, default='VGG16', \
            help='Model\n'
                 '  - MobileNetv2 and CenterNetHourGlass104 does not work')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='config', \
            help='Output directory')

    args = parser.parse_args()

    return args


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
    else:
        assert f'{args.model} is not supported'
    
    model.summary()
    
    for i, layer in enumerate(model.layers):
        print(f'layer{i} config: {layer.get_config()}')
        if (layer.__class__.__name__ == 'Conv2D'):
            config = {
                'model': args.model,
                'index': i,
                'name': layer.name,
            }
            
            config_path = Path(args.output_dir, f'layer{i}_{layer.name}.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            
if __name__=='__main__':
    main()
