# Create Custom Dataset Sample

This tool is the sample program to create the custom dataset for AI Dashboard.

## Executable Environment

This tool can be run on "web" service in AI Dashboard services.

```
$ cd AI_Dashboard
$ docker-compose exec web bash
```

## Usage

```
$ python create_custom_dataset_sample.py --help
usage: create_custom_dataset_sample.py [-h] [--dataset_name DATASET_NAME] [--input_dir INPUT_DIR]
                                       [--output_dir OUTPUT_DIR] [--n_data N_DATA]

This tool convert from open dataset to AI Dashboard's custom dataset format

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        specifies the open dataset name (cifar-10 or mnist)
  --input_dir INPUT_DIR
                        input directory if necessary the download
  --output_dir OUTPUT_DIR
                        output directory
  --n_data N_DATA       number of data samples (if set to less than 0, get all samples)
```

## Example

### MNIST

```
# python create_custom_dataset_sample.py --dataset_name mnist --output_dir output_mnist --n_data 100 --validation_split 0.2
```

### CIFAR-10

```
# python create_custom_dataset_sample.py --dataset_name cifar-10 --output_dir output_cifar10 --n_data 100 --validation_split 0.2
```

