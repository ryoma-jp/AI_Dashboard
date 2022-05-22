
import time
import numpy as np
import tflite_runtime.interpreter as tflite

def _infrerence(input_details, output_details, input_data):
    '''Inference (internal function)
    TensorFlow Lite inference

    Arguments:
      - input_details: result of tflite.Interpreter.get_input_details()
      - output_details: result of tflite.Interpreter.get_output_details()
      - input_data: input inference data

    Returns:
      - output_data: result of inference
      - time: inference time [ms]
    '''
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    stop_time = time.time()

    time = (stop_time - start_time) * 1000
    return output_data, time

def object_detection(model_file, image):
    '''Object Detection

    Arguments:
      - model_file: file path of tflite
      - image: image data(ndarray, HxWxC)
    '''

    # load model file
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # image preprocessing [T.B.D]

    # inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    result, time = _inference(input_details, output_details, image)

    # draw bounding box [T.B.D]

    # return
    ret_image = image
    return ret_image

