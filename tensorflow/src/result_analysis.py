import itertools
import os
import csv
import json
import pickle
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from itertools import combinations
from constants import *


def normalize_pred(pred: Tensor):
    pred = tf.nn.softmax(pred)
    return pred


def read_result(model: int, idx: int, rtol, atol, rtol_2, atol_2):
    result_path = f"../results/outputs/output_{model}/"

    preds = []
    for c in configs:
        # check if frozen_order.txt exists
        if os.path.exists(f"../data/models/model_{model}/frozen_order.txt"):
            c += "_f1"
        try:
            with open(result_path + f"output_{idx}_{c}.pk", 'rb') as f:
                pred = pickle.load(f)
                preds.append(pred)
        except Exception as e:
            print(f"Error reading {model} {idx} {c}", e)
            preds.append(None)

    diffs = {}
    closes = []

    for i in range(len(preds)):
        if preds[i] is not None:
            preds[i] = tf.nn.softmax(preds[i])

    for i in range(len(preds)):
        configs_i = configs[i]
        configs_base = "0" + configs_i[1:]  # 0 is the base config
        index_base = configs.index(configs_base)
        if index_base == -1 or index_base == i:
            closes.append(True)
            continue
        pred_1, pred_2 = preds[index_base], preds[i]
        if pred_1 is None or pred_2 is None:
            closes.append(True)
            continue

        max_diff = np.max(np.abs(pred_1 - pred_2))
        diffs[configs[i]] = float(max_diff)

        closes.append(np.allclose(pred_1, pred_2, rtol=rtol, atol=atol))

    return np.array(closes), diffs


def get_result_for_input(input_idx: int):
    closes, diffs = read_result(
        model_idx, input_idx, rtol=1e-4, atol=5e-4, rtol_2=1e-4, atol_2=5e-4)
    
    n_inconsistency = sum(1 for x in closes if x == False)

    if n_inconsistency == 0:
        return False

    print(f"Model {model_idx} input {input_idx} inconsistency {n_inconsistency}, max diff: {max(diffs.values())}")
    print("Inconsistent settings: ", end="")
    for (i, c) in enumerate(closes):
        if c == False:
            inconsistency_list.append({
                "model": model_idx,
                "input": input_idx,
                "config": configs[i],
                "Linf": diffs[configs[i]]
            })
            print(configs[i], end=", ")
    print()

    global global_inconsistency
    global_inconsistency += n_inconsistency
    return True


def get_result_for_model(model_idx: int):
    model_inconsist = False
    for input_idx in range(NUM_TRAINING_DATASETS):
        try:
            model_inconsist |= get_result_for_input(input_idx)
        except Exception as e:
            failed_models.add(model_idx)
            print(model_idx, e)
            # continue

    if model_inconsist:
        inconsist_models.append(model_idx)
        global n_inconsist_model
        n_inconsist_model += 1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global_inconsistency = 0
    n_inconsist_model = 0

    device_type_lst = ["GPU", "CPU"]
    num_device_lst = [0, 1, 2, 3, 4, 8]
    extra_lst = [None, "quantize_8_8"]
    configs = []
    for num_device, device_type, model_extra in itertools.product(num_device_lst, device_type_lst, extra_lst):
        dist_setting_str = str(num_device) + device_type
        suffix = ""
        if model_extra in ["quantize_8_8"]:
            suffix += f"_{model_extra}"
        config = f"{dist_setting_str}{suffix}"
        configs.append(config)


    inconsist_models = []
    failed_models = set()
    inconsistency_list = []

    headers = ["model", "input", "config", "Linf"]

    for model_idx in range(NUM_MODELS):
        get_result_for_model(model_idx)

    print("Failed models:", len(failed_models))
    print(failed_models)
    print("Model with inconsistencies:", n_inconsist_model)
    print(inconsist_models)
    print("Total inconsistencies:", global_inconsistency)

    with open("../results/csv/inconsistency.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(inconsistency_list)
