import os
import argparse
import json

from pathlib import Path
from tensorflow.keras.applications import VGG16

def ArgParser():
    parser = argparse.ArgumentParser(description='Create config file for tf-keras-vis application',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--model', dest='model', type=str, choices=['VGG16'], default='VGG16', \
            help='Model')
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
    
    for i, layer in enumerate(model.layers):
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
