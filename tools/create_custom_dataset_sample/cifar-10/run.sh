#! /bin/bash

# --- parameters ---
INPUT_DIR="input"
OUTPUT_DIR="output"
N_DATA=0		# get all data

# --- create directories ---
mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# --- download CIFAR-10 dataset ---
BASE_DIR=${PWD}
cd ${INPUT_DIR}
dataset_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
wget ${dataset_url}
tar -zxf cifar-10-python.tar.gz
cd ${BASE_DIR}

# --- preprocessing ---
python main.py --input_dir ${INPUT_DIR}/cifar-10-batches-py --output_dir ${OUTPUT_DIR} --n_data ${N_DATA}

