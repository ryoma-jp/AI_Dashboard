"""Main routine of Machine Learning

Training and Testing models with AI Model SDK.
"""

#---------------------------------
# Import modules
#---------------------------------
import os
import sys
import argparse

#---------------------------------
# Functions
#---------------------------------
def ArgParser():
    """ArgParser
    
    Load arguments
    
    """
    
    parser = argparse.ArgumentParser(description='Training and Testing with AI Model SDK',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--sdk_path', dest='sdk_path', type=str, default=None, required=True, \
            help='AI Model SDK path')
    parser.add_argument('--meta_json', dest='meta_json', type=str, default=None, required=True, \
            help='info.json for meta data')
    parser.add_argument('--train_json', dest='train_json', type=str, default=None, required=False, \
            help='info.json for train data')
    parser.add_argument('--val_json', dest='val_json', type=str, default=None, required=False, \
            help='info.json for val data')
    parser.add_argument('--test_json', dest='test_json', type=str, default=None, required=False, \
            help='info.json for test data')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None, required=True, \
            help='AI Model Path')

    args = parser.parse_args()

    return args

def main():
    """main
    
    Main function
    
    """
    args = ArgParser()
    print('[INFO] Arguments')
    print('  * args.sdk_path = {}'.format(args.sdk_path))
    print('  * args.meta_json = {}'.format(args.meta_json))
    print('  * args.train_json = {}'.format(args.train_json))
    print('  * args.val_json = {}'.format(args.val_json))
    print('  * args.test_json = {}'.format(args.test_json))
    print('  * args.model_path = {}'.format(args.model_path))
    
    # --- add AI Model SDK path to Python path ---
    sys.path.append(args.sdk_path)
    
    # --- import AI Model SDK ---
    from ai_model_sdk import AI_Model_SDK
    print(AI_Model_SDK.__version__)

    # --- Create instance ---
    dataset_params = {
        'meta': args.meta_json,
        'train': args.train_json,
        'val': args.val_json,
        'test': args.test_json,
    }
    model_params = {
        'model_path': args.model_path,
    }
    ai_model_sdk = AI_Model_SDK(dataset_params, model_params)

    # --- load dataset ---
    ai_model_sdk.load_dataset()

    # --- build model ---
    ai_model_sdk.build_model()

    # --- training ---
    ai_model_sdk.train_model()

    return

#---------------------------------
# main routine
#---------------------------------
if __name__ == '__main__':
    main()

