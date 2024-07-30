import os
import csv
import json
import pickle
import keras
import numpy as np
import tensorflow as tf
import traceback
from typing import *
from tensorflow import Tensor
from constants import NUM_MODELS
import tensorflow_model_optimization as tfmot


def normalize_pred(pred: Tensor):
    pred = tf.nn.softmax(pred)
    return pred


def calc_MAD(y: Tensor, o: Tensor) -> float:
    # y: predicted value
    # o: ground truth
    return np.sum(np.abs(y - o)) / len(y)


def calc_RL(m: float, m_p: float) -> float:
    # m: mad of current layer
    # m_p: maximum mad up to current layer
    return (m - m_p) / (m_p + 1e-7)


def get_layer_mapping(model1: keras.Model, model2: keras.Model):
    model1.summary()
    model2.summary()
    layers1 = model1.layers
    layers2 = model2.layers
    for (i, l) in enumerate(layers2):
        pass


def analyze_layer_results(m: int, i: int, s1: str, s2: str):
    # m: model idx, i: input idx, s1: setting1, s2: setting2
    result_path = f"../results/analysis/output_{m}_{i}/"
    model_path = f"../data/models/model_{m}/"

    layer = 0
    diff_lst = []
    mad_lst = []
    RL_lst = []

    if "_q" in s1 or "_q" in s2:
        with tfmot.quantization.keras.quantize_scope():
            model: keras.Model = keras.models.load_model(model_path + "model_quantized.h5")
    else:
        model: keras.Model = keras.models.load_model(model_path + "model.h5")
    model.summary()
    # layer_mapping = get_layer_mapping(model_orig, model)

    with open(result_path + f"output_{s1}.pk", 'rb') as f:
        preds_1 = pickle.load(f)

    with open(result_path + f"output_{s2}.pk", 'rb') as f:
        preds_2 = pickle.load(f)

    for layer in range(len(preds_1)):
        pred_1 = normalize_pred(preds_1[layer])
        pred_2 = normalize_pred(preds_2[layer])

        # Calculate difference
        max_diff = np.max(np.abs(pred_1 - pred_2))
        diff_lst.append(max_diff)

        # Calculate MAD, assume the first setting is the ground truth
        mad = calc_MAD(pred_2, pred_1)

        # Get the model structure
        # with open(model_path + "model.json", 'r') as f:
        #     model_structure = json.load(f)["model_structure"]

        # print("Layer", layer)
        layer_x = model.layers[layer]
        # print(len(layer_x._inbound_nodes))
        int_node = layer_x._inbound_nodes[0]
        if layer == 0:
            pre_layers = []
        elif type(int_node.inbound_layers) is list:
            num_predecessor = len(int_node.inbound_layers)
            pre_layers = [model.layers.index(int_node.inbound_layers[i]) for i in range(num_predecessor)]
        else:
            pre_layers = [model.layers.index(int_node.inbound_layers)]

        # print(pre_layers)

        # pre_layers = model_structure[str(layer)]["pre_layers"]
        if len(pre_layers) == 0:
            RL_lst.append(calc_RL(mad, 0))
        else:
            max_mad = max(mad_lst[i] for i in pre_layers)
            RL_lst.append(calc_RL(mad, max_mad))

        mad_lst.append(mad)

    print("Linf:", diff_lst)
    print("MAD:", mad_lst)
    print("RL:", RL_lst)

    # print(f"Max Linf {np.max(diff_lst)} at layer {np.argmax(diff_lst)}")
    # print(f"Max mad {np.max(mad_lst)} at layer {np.argmax(mad_lst)}")

    idx_max_RL = np.argmax(RL_lst)
    layer_1_n = model.layers[idx_max_RL].name
    if "quantize_layer" in layer_1_n:
        layer_1_t = "quantize_layer"
    else:
        layer_1_t = '_'.join(model.layers[idx_max_RL].name.replace("quant_", "").split('_')[1:])

    idx_second_RL = np.argsort(RL_lst)[-2]
    layer_2_n = model.layers[idx_second_RL].name
    if "quantize_layer" in layer_2_n:
        layer_2_t = "quantize_layer"
    else:
        layer_2_t = '_'.join(model.layers[idx_second_RL].name.replace("quant_", "").split('_')[1:])

    print(
        # f"Max RL {RL_lst[idx_max_RL]} at layer {idx_max_RL}: {model_structure[str(idx_max_RL)]['type']}")
        f"Max RL {RL_lst[idx_max_RL]} at layer {idx_max_RL}: {layer_1_n}")
    print(
        f"Second largest RL {RL_lst[idx_second_RL]} at layer {idx_second_RL}: {layer_2_n}")
    print()

    row = {"Model_id": m,
           "Input_id": i,
           "Setting_1": s1,
           "Setting_2": s2,
           "Linf_output": diff_lst[-1],

           "Idx_max_RL": idx_max_RL,
           "Layer_1": layer_1_n,
           "Layer_1_t": layer_1_t,
           "RL_1": RL_lst[idx_max_RL],
           "MAD_1": mad_lst[idx_max_RL],
           "Linf_1": diff_lst[idx_max_RL],

           "Idx_second_RL": idx_second_RL,
           "Layer_2": layer_2_n,
           "Layer_2_t": layer_2_t,
           "RL_2": RL_lst[idx_second_RL],
           "MAD_2": mad_lst[idx_second_RL],
           "Linf_2": diff_lst[idx_second_RL]
           }

    return row


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    csv_header = ["Model_id", "Input_id", "Setting_1", "Setting_2", "Linf_output", "Idx_max_RL", "Layer_1",
                  "Layer_1_t", "RL_1", "MAD_1", "Linf_1", "Idx_second_RL", "Layer_2", "Layer_2_t", "RL_2", "MAD_2", "Linf_2"]
    rows = []

    with open("../results/csv/inconsistency.csv", 'r') as f:
        inconsistencies = f.readlines()

    model_id = -1
    for line in inconsistencies[1:]:
        model_id, input_id, setting, Linf = line.strip().split(',')
        model_id = int(model_id)
        input_id = int(input_id)
        if model_id < NUM_MODELS // 2:
            continue
        print(model_id, input_id, setting)

        s2 = setting
        s1 = "0" + s2[1:]

        try:
            print(f"Setting: {s1} vs {s2}")
            rows.append(analyze_layer_results(model_id, input_id, s1, s2))

        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            # break

    with open("../results/csv/cluster.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(rows)

