
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def _inference(interpreter, input_details, input_data):
    '''Inference (internal function)
    TensorFlow Lite inference
    the inference result can be get to use interpreter.get_tensor

    Arguments:
      - interpreter: tflite.Interpreter
      - input_details: result of tflite.Interpreter.get_input_details()
      - input_data: input inference data

    Returns:
      - duration: inference time [ms]
    '''
    start_time = time.time()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    stop_time = time.time()

    duration = (stop_time - start_time) * 1000
    return duration

def object_detection(model_file, class_label, image, max_detection=-1, thresh=0.3):
    '''Object Detection

    Arguments:
      - model_file: file path of tflite
      - class_label: list of class name
      - image: image data(ndarray, HxWxC)
      - max_detection: max boxes to draw on image (if -1 set then draw all boxes, but thresh is more priority)
      - thresh: minimum score threshold to draw boxes

    Returns:
      - ret_image: return image
      - duration: inference time [ms]
    '''

    # load model file
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    # image preprocessing
    org_img_shape = image.shape
    input_img_shape = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    input_image = cv2.resize(image, input_img_shape)
    input_image = np.expand_dims(input_image, axis=0)

    # inference
    duration = _inference(interpreter, input_details, input_image)

    # draw bounding box
    output_details = interpreter.get_output_details()
        # StatefulPartitionedCall:0  count
        # StatefulPartitionedCall:1  scores
        # StatefulPartitionedCall:2  classes
        # StatefulPartitionedCall:3  boxes

    result = {}
    for output_detail in output_details:
        if (output_detail['name'] == 'StatefulPartitionedCall:0'):
            result['count'] = interpreter.get_tensor(output_detail['index'])
        elif (output_detail['name'] == 'StatefulPartitionedCall:1'):
            result['scores'] = interpreter.get_tensor(output_detail['index'])
        elif (output_detail['name'] == 'StatefulPartitionedCall:2'):
            result['classes'] = interpreter.get_tensor(output_detail['index'])
        elif (output_detail['name'] == 'StatefulPartitionedCall:3'):
            result['boxes'] = interpreter.get_tensor(output_detail['index'])

    ret_image = image
    for i, (score_, class_, box_) in enumerate(zip(result['scores'][0], result['classes'][0], result['boxes'][0])):
        if (score_ >= thresh):
            print(f'{i}, score={score_}, class={class_}, box={box_}')

            ymin, xmin, ymax, xmax = box_
            ymin = int(ymin * org_img_shape[0])
            xmin = int(xmin * org_img_shape[1])
            ymax = int(ymax * org_img_shape[0])
            xmax = int(xmax * org_img_shape[1])

            color = [255, 0, 0]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            y = ymin - 15 if ((ymin - 15) > 15) else ymin + 15
            label = f'{class_label[int(class_)]}'
            cv2.putText(image, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # return
    return ret_image, duration

