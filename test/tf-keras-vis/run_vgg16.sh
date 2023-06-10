#! /bin/bash

model="VGG16"
config_dir="config_${model}"

python3 create_config.py --model ${model} --output_dir ${config_dir}

files="${config_dir}/*"
vis_dir="vis_result_${model}"
for file in ${files}; do
    echo ${file}
    python3 tf-keras-vis-app.py --config ${file} --output_dir ${vis_dir}
done


