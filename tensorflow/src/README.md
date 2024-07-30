# D3 TensorFlow

This is the repo for reproducing the TensorFlow experiment in D3: Differential Testing of Distributed Deep Learning with Model Generation. Note that the following command should be executed while in folder `/path/to/D3/tensorflow/src`.

## 1. Setup folders

```shell
chmod +x ./setup.sh && ./setup.sh
```

## 2. Setup Muffin docker

```shell
docker pull librarytesting/muffin:E1
docker run --runtime=nvidia -it -v $PWD/muffin:/data --name muffin librarytesting/muffin:E1 /bin/bash
```

(now inside muffin docker)
```shell
source activate lemon
cd /data/dataset
python get_dataset.py cifar10
```

Now you can exit the Muffin docker.

## 3. Setup TensorFlow docker

```shell
docker pull tensorflow/tensorflow:devel-gpu
docker run -it --name D3-tf --gpus all -v "$PWD":"/mnt" -v "$PWD/../data":"/data" -v "$PWD/../results":"/results" -w "/mnt" tensorflow/tensorflow:devel-gpu bash
```

(now inside TensorFlow docker)
```shell
pip install tensorflow==2.11.0 tensorflow_addons==0.19.0 tensorflow-model-optimization==0.7.3
```

Now you can exit the TensorFlow docker.

## 4. Generate models and inputs

You can control the batch size, number of models and number of inputs generated in `constants.py`. 
The default values in this demo are: `BATCH_SIZE=2400, NUM_MODELS=8, NUM_TRAINING_DATASETS=1`.
To reproduce the results in the paper, use `BATCH_SIZE=2400, NUM_MODELS=400, NUM_TRAINING_DATASETS=10`.

If the dockers are not running, start the dockers first.
```shell
docker start muffin
docker start D3-tf
```

```shell
chmod 777 ./muffin/data
./generate_muffin.sh muffin
```

**All the following commands should be executed inside the TensorFlow docker environment**
```shell
docker exec -u $(id -u):$(id -g) -it D3-tf bash
```

Inside the Tensorflow docker:
```shell
python rename_muffin.py
python gen_model_and_input.py
python get_freeze_order.py
python convert_to_quantized_with_datatype.py
```

## 5. Run the experiment

You can change the constants in `run_multiple_process.py` to run less settings. Default behavior is to run all settings.

```shell
python run_multiple_process.py
```

## 6. Result analysis
Generate a csv file including all the inconsistencies
```shell
python result_analysis.py
```

