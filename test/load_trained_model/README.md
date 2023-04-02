# Sample program of loading trained model is trained by AI_Dashboard

## Usage

```
$ python3 load_trained_model.py -h
2023-04-02 01:00:22.664694: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
usage: load_trained_model.py [-h] --trained_model TRAINED_MODEL

Sample program for loading trained model

optional arguments:
  -h, --help            show this help message and exit
  --trained_model TRAINED_MODEL
                        Path for trained model path
                          - if specify the saved_model, set path to saved_model directory
                          - if specify the h5, set path to h5 file
```

## Exec sample

### Prepare

```shell
$ cd AI_Dashboard
$ ./ build_and_run.sh
$ docker-compose exec web bash
$ cd /home/app/web/test/load_trained_model
```

### Execute
```
$ TRAINED_MODEL="../../../media/model/<project_hash>/<model_hash>/models/saved_model/"
$ python3 load_trained_model.py --trained_model $TRAINED_MODEL
```

### Execution sample for loading saved_model

```
$ TRAINED_MODEL="../../../media/model/5c6bed0d94b9be8afbc5c8cac1e9d4be03f556917c2611ec56f4e6f341ef60d9/84d1c87c71f48569b0d50bf4e3cf3ea8e06a2faa1f949b8391d82848b69489e5/models/saved_model/"
$ python3 load_trained_model.py --trained_model $TRAINED_MODEL
```

### Execution sample for loading h5

```
$ TRAINED_MODEL="../../../media/model/5c6bed0d94b9be8afbc5c8cac1e9d4be03f556917c2611ec56f4e6f341ef60d9/84d1c87c71f48569b0d50bf4e3cf3ea8e06a2faa1f949b8391d82848b69489e5/models/hdf5/model.h5"
$ python3 load_trained_model.py --trained_model $TRAINED_MODEL
```

