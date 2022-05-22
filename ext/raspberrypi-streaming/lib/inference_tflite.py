
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def _inference(interpreter, input_details, output_details, input_data):
    '''Inference (internal function)
    TensorFlow Lite inference

    Arguments:
      - interpreter: tflite.Interpreter
      - input_details: result of tflite.Interpreter.get_input_details()
      - output_details: result of tflite.Interpreter.get_output_details()
      - input_data: input inference data

    Returns:
      - output_data: result of inference
      - duration: inference time [ms]
    '''
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    stop_time = time.time()

    duration = (stop_time - start_time) * 1000
    return output_data, duration

def object_detection(model_file, image):
    '''Object Detection

    Arguments:
      - model_file: file path of tflite
      - image: image data(ndarray, HxWxC)

    Returns:
      - ret_image: return image
      - duration: inference time [ms]
    '''

    # for debug
    ret_image = image

    # load model file
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # image preprocessing [T.B.D]
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    image = cv2.resize(image, (height, width))
    image = np.expand_dims(image, axis=0)

    # inference
    result, duration = _inference(interpreter, input_details, output_details, image)

    # draw bounding box [T.B.D]

    # return
    return ret_image, duration

