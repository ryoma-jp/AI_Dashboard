#! /bin/bash

# --- Get image from YouTube ---

# --- Load CenterNetHourGlass104 model and Prediction
python << 'EOT'
#from machine_learning.lib.predictor.predictor_keras import PredictorCenterNetHourGlass104
#
#model = PredictorCenterNetHourGlass104()
#model.pretrained_model.summary()

import tensorflow as tf
import tensorflow_hub as hub

hourglass = False
if hourglass:
    # --- ERROR Occurred ---
    #        raise ValueError(SINGLE_LAYER_OUTPUT_ERROR_MSG)
    #    ValueError: All layers in a Sequential model should have a single output tensor.
    #    For multi-output layers, use the functional API.
    model_url = 'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'
    image_shape = (512, 512)
    dtype = tf.uint8
else:
    model_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    image_shape = (224, 224)
    dtype = tf.float32

model_sequential = tf.keras.Sequential([
    hub.KerasLayer(model_url, input_shape=image_shape+(3,), dtype=dtype)
])

model_sequential.summary()

for i, w in enumerate(model_sequential.layers[0].weights):
    print(f'Layer #{i}: {w.shape}')

EOT

