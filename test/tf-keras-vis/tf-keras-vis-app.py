import os
import argparse
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications import VGG16
from matplotlib import pyplot as plt
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D
from tf_keras_vis.activation_maximization.regularizers import TotalVariation2D, Norm
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayer, ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from machine_learning.lib.trainer.tf_models.yolov3 import models as yolov3_models

FILTER_INDEX = 0

def ArgParser():
    parser = argparse.ArgumentParser(description='Sample application for tf-keras-vis',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', dest='config', type=str, required=True, \
            help='Config file path(*.json)\n'
                 '  {\n'
                 '    "model": <model name(VGG16, YOLOv3)>\n'
                 '       - YOLOv3 does not work\n'
                 '    "index": <layer index(0, 1, ...)>\n'
                 '    "name": <layer name>\n'
                 '  }')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='config', \
            help='Output directory')

    args = parser.parse_args()

    return args


def main():
    args = ArgParser()
    print(f'args.config: {args.config}')
    print(f'args.output_dir: {args.output_dir}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        dict_config = json.load(f)
    
    model_name = dict_config['model']
    layer_index = dict_config['index']
    layer_name = dict_config['name']
    
    if (model_name == 'VGG16'):
        model = VGG16()
    elif (model_name == 'YOLOv3'):
        model = yolov3_models.YoloV3(size=416, classes=80, training=True)
    else:
        assert f'{model_name} is not supported'
    
    # Create the visualization instance.
    # All visualization classes accept a model and model-modifier, which, for example,
    #     replaces the activation of last layer to linear function so on, in constructor.
    activation_maximization = \
       ActivationMaximization(model,
                              model_modifier=[ExtractIntermediateLayer(layer_name),
                                              ReplaceToLinear()],
                              clone=False)

    # You can use Score class to specify visualizing target you want.
    # And add regularizers or input-modifiers as needed.
    activations = \
       activation_maximization(CategoricalScore(FILTER_INDEX),
                               steps=200,
                               input_modifiers=[Jitter(jitter=16), Rotate2D(degree=1)],
                               regularizers=[TotalVariation2D(weight=1.0),
                                             Norm(weight=0.3, p=1)],
                               optimizer=tf.keras.optimizers.RMSprop(1.0, 0.999),
                               callbacks=[Progress()])

    ## Since v0.6.0, calling `astype()` is NOT necessary.
    # activations = activations[0].astype(np.uint8)

    # save image
    activation = activations[0].numpy().astype(dtype=np.uint8)
    #print(type(activations[0]))
    #print(type(activation))
    #print(activation.min())
    #print(activation.max())
    save_file = Path(args.output_dir, f'activations_layer{layer_index}_{layer_name}.png')
    plt.imsave(save_file, activation)

if __name__=='__main__':
    main()
