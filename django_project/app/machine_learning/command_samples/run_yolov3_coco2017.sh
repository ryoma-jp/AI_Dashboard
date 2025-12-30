#! /bin/bash

dataset_dir="<path-to-dataset>"
dataset_type="COCO2017"
sdk_path="/home/app/media/ai_model_sdk/sample_sdk/YOLOv3_for_COCO2017/"
dataset="${dataset_dir}/dataset.pkl"
meta_info="${dataset_dir}/meta/info.json"
train_info="${dataset_dir}/train/info.json"
val_info="${dataset_dir}/validation/info.json"
test_info="${dataset_dir}/test/info.json"
model_path="./tmp_output"

#python3 ml_train_main.py \
#	--sdk_path ${sdk_path} \
#	--dataset ${dataset} \
#	--meta_json ${meta_info} \
#	--train_json ${train_info} \
#	--val_json ${val_info} \
#	--test_json ${test_info} \
#	--model_path ${model_path}

python3 ml_train_main.py \
	--sdk_path ${sdk_path} \
	--dataset_type ${dataset_type} \
	--model_path ${model_path}
