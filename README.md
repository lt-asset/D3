# D3

This is the repo for a demo to reproduce the experiment of D3: Differential Testing of Distributed Deep Learning with Model Generation.

## D3 TensorFlow

To run the TensorFlow experiment, first do `cd tensorflow/src` then follow the instructions in the [README.md](./tensorflow/src/README.md).

It will first generates models and inputs, then runs the experiment for distributed settings, and finally analyzes the results and generate a csv file including all the inconsistencies.

We use prior work muffin [[1]](#1) to generate sequential models. We include a modified version of Muffin in the repo ([link](./tensorflow/src/muffin/)).

We use prior work from this repo [[2]](#2) to build DLRM-like models in TensorFlow.

We recommend running TensorFlow experiment first because the PyTorch experiment needs to convert the sequential models generated by Muffin from TensorFlow to PyTorch.

## D3 PyTorch

To run the PyTorch experiment, first do `cd pytorch/src` then follow the instructions in the [README.md](./pytorch/src/README.md).

It will first generates or converts models and inputs, then runs the experiment for distributed settings, and finally analyzes the results and generates a csv file including all the inconsistencies.

## References
<a id="1">[1]</a> 
Gu Jiazhen, Luo Xuchuan, Zhou Yangfan, Wang Xin.
Muffin: Testing Deep Learning Libraries via Neural Architecture Fuzzing. 
2022 IEEE/ACM 44th International Conference on Software Engineering (ICSE).

<a id="2">[2]</a> 
https://github.com/NodLabs/tensorflow-dlrm/tree/master
