
import argparse
import tensorflow as tf

def arg_parser():
    parser = argparse.ArgumentParser(description='Sample program for loading trained model',
                formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--trained_model', dest='trained_model', type=str, default=None, required=True, \
            help='Path for trained model path \n' \
                 '  - if specify the saved_model, set path to saved_model directory \n' \
                 '  - if specify the h5, set path to h5 file')

    args = parser.parse_args()

    return args

def main():
    args = arg_parser()
    print(f'args.trained_model : {args.trained_model}')
    
    model = tf.keras.models.load_model(args.trained_model)
    model.summary()

if __name__ == '__main__':
    main()
    