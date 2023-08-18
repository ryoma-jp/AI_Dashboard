"""Main routine of Machine Learning for Inference

Inference data using by the trained model with AI Model SDK.
"""

#---------------------------------
# Import modules
#---------------------------------
import os
import sys
import argparse
from pathlib import Path

#---------------------------------
# Functions
#---------------------------------
def ArgParser():
    """ArgParser
    
    Load arguments
    
    """
    
    parser = argparse.ArgumentParser(description='Training with AI Model SDK',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--sdk_path', dest='sdk_path', type=str, default=None, required=True, \
            help='AI Model SDK path')
    parser.add_argument('--meta_json', dest='meta_json', type=str, default=None, required=True, \
            help='info.json for meta data')
    parser.add_argument('--inference_json', dest='inference_json', type=str, default=None, required=False, \
            help='info.json for inference data')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None, required=True, \
            help='AI Model Path')
    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, required=True, \
            help='Trained Model Path')

    args = parser.parse_args()

    return args

def main():
    """main
    
    Main function
    """
    args = ArgParser()
    print('[INFO] Arguments')
    print(f'  * args.sdk_path = {args.sdk_path}')
    print(f'  * args.meta_json = {args.meta_json}')
    print(f'  * args.inference_json = {args.inference_json}')
    print(f'  * args.model_path = {args.model_path}')
    print(f'  * args.trained_model_path = {args.trained_model_path}')
    
    # --- add AI Model SDK path to Python path ---
    sys.path.append(args.sdk_path)
    
    # --- import AI Model SDK ---
    from ai_model_sdk import AI_Model_SDK
    print(AI_Model_SDK.__version__)

    # --- Create instance ---
    dataset_params = {
        'meta': args.meta_json,
        'inference': args.inference_json,
    }
    model_params = {
        'model_path': args.model_path,
    }
    ai_model_sdk = AI_Model_SDK(dataset_params, model_params)

    # --- load dataset ---
    ai_model_sdk.load_dataset()

    # --- load model ---
    ai_model_sdk.load_model(args.trained_model_path)

    # --- inference ---
    prediction = ai_model_sdk.predict(ai_model_sdk.x_inference)
    print(prediction.shape)
    print(prediction)

    return

#---------------------------------
# main routine
#---------------------------------
if __name__ == '__main__':
    main()

