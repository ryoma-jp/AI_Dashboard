#! /bin/bash

# --- parameters ---
INPUT_DIR="input"
OUTPUT_DIR="output"
N_DATA=0		# get all data

# --- create directories ---
mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# --- download MNIST dataset ---
BASE_DIR=${PWD}
cd ${INPUT_DIR}
dataset_url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget ${dataset_url}
gunzip train-images-idx3-ubyte.gz

dataset_url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
wget ${dataset_url}
gunzip train-labels-idx1-ubyte.gz

dataset_url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
wget ${dataset_url}
gunzip t10k-images-idx3-ubyte.gz

dataset_url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
wget ${dataset_url}
gunzip t10k-labels-idx1-ubyte.gz

cd ${BASE_DIR}

# --- preprocessing ---
python main.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --n_data ${N_DATA}

# --- zip compress ---
cd ${OUTPUT_DIR}
zip -rq train_data.zip train_data
zip -rq test_data.zip test_data

