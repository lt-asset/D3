# D3 PyTorch

This is the repo for reproducing the PyTorch experiment in D3: Differential Testing of Distributed Deep Learning with Model Generation. Note that the following command should be executed while in folder `/path/to/D3/pytorch/src`.


## 1. Setup PyTorch docker

```shell
docker pull nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
docker run -it --name D3-pt --gpus all -v "$PWD":"/mnt" -v "$PWD/../data":"/data" -v "$PWD/../../tensorflow/data/models":"/tensorflow/data/models" -v "$PWD/../results":"/results" -w "/mnt" --ipc=host --network host nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
```

(now inside PyTorch docker)
We install conda inside the docker.
```shell
apt-get update
apt-get install -y curl
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/root/miniconda3/bin:$PATH"
rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh

conda init bash
source ~/.bashrc
conda deactivate
```

## **All the following commands should be executed inside the PyTorch docker environment**

## 2. Generate models and inputs

You can control the batch size, number of models and number of inputs generated in `constants.py`. 
The default values in this demo are: `BATCH_SIZE=2400, NUM_MODELS=8, NUM_TRAINING_DATASETS=1`.
To reproduce the results in the paper, use `BATCH_SIZE=2400, NUM_MODELS=400, NUM_TRAINING_DATASETS=10`.

For the sequential models, we create a tf2onnx virtual environment.

```shell
conda create -y -n tf2onnx python=3.8
conda activate tf2onnx
pip install tensorflow
pip install -U tf2onnx
```

Then we convert the sequential models to PyTorch.

```shell
python convert_tf_2_onnx.py
```

Finally we close the virtual enviroment.

```shell
conda deactivate
```

For the DLRM-like models, we create a PyTorch virtual enviroment.

```shell
conda create -y -n pt112tr02 python=3.8
conda activate pt112tr02
pip install onnx2pytorch==0.4.1
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchrec==0.2.0 fbgemm-gpu==0.2.0
pip install chardet
```

Then we generate the DLRM-like models and input to both DLRM-like models or sequential models.

```shell
python gen_model_and_input.py
```

## 3. Run the experiment 

First run the multiple process script to run all the non-quantization settings, then run the single process script to run all the quantization settings.
You can change the constants in `config.py` to run less settings. Default behavior is to run all settings.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_multiple_process.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_single_process.py
```

## 4. Result analysis

Generate a csv file including all the inconsistencies.

```shell
python result_analysis.py
```
