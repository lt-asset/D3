import argparse
import tensorflow as tf
import keras
import pickle
import random
import os
import sys
import numpy as np
import itertools
from keras import Sequential, optimizers, losses
from keras import backend as K
from constants import NUM_MODELS
from test_utils import generateStrategy
import tensorflow_model_optimization as tfmot
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()


def normalize_pred(pred):
    pred = tf.nn.softmax(pred)
    return pred


def layer_output(model_id: int, input_id: int, device: str):
    model_dir = f"../data/models/model_{model_id}/"
    input_dir = f"../data/inputs/input_{model_id}/"
    out_dir = f"../results/analysis/output_{model_id}_{input_id}/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + f"{device}.txt"

    num_device = int(device[0])
    type_device = device[1:4]
    model_extra = device[5:].strip("_f1")
    strategy = generateStrategy(
        num_device=num_device, strategy=tf.distribute.MirroredStrategy, device_type=type_device)
    with strategy.scope():
        if model_extra == "q":
            with tfmot.quantization.keras.quantize_scope():
                model = keras.models.load_model(
                    model_dir + "model_quantized.h5")
        elif model_extra in ["quantize_8_8", "quantize_8_16", "quantize_16_8", "quantize_16_16"]:
            num_bits_weight = int(model_extra.split("_")[1])
            num_bits_activation = int(model_extra.split("_")[2])
            with tfmot.quantization.keras.quantize_scope():
                model = keras.models.load_model(
                    model_dir + f"model_quantized_{num_bits_weight}_{num_bits_activation}.h5")
        else:
            model = keras.models.load_model(
                model_dir + "model.h5")
        
        if os.path.exists(model_dir + "frozen_order.txt") and "f1" in device:
            with open(model_dir + "frozen_order.txt", 'r') as f:
                frozen_order = eval(f.read())
            for i in range(min(len(frozen_order), 1)):
                model.layers[frozen_order[i]].trainable = False


        optimizer = optimizers.SGD(learning_rate=10)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.AUTO)
        model.compile(optimizer=optimizer, loss=loss)

    # input placeholder
    inp = model.input
    # all layer outputs
    outputs = [layer.output for layer in model.layers]
    # evaluation function
    functors = [K.function([inp], [out]) for out in outputs]

    with open(input_dir + f"train_{input_id}.pk", 'rb') as f:
        (train_input, train_label) = pickle.load(f)
    model.fit(train_input, train_label, verbose=0, shuffle=False, batch_size=2400)

    with open(input_dir + f"test_{input_id}.pk", 'rb') as f:
        (test_input, _) = pickle.load(f)

    data_out = []

    with open(out_file, 'w') as f:
        # Testing
        test = test_input
        layer = 0
        for func in functors:
            f.write(f"Layer {layer} Name: {model.layers[layer]}")
            f.write('\n')
            res = func([test])
            f.write(f"{res}")
            f.write('\n')

            # Save output
            data_out.append(np.array(res).squeeze())
            layer += 1

        after_normalize = normalize_pred(data_out[-1])
        f.write("After normalization:\n")
        f.write(f"{after_normalize}")

    with open(f"{out_dir}/output_{device}.pk", 'wb') as f:
        pickle.dump(data_out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["CPU", "GPU"],default="CPU")
    parser.add_argument("--infile", type=str, default="../results/csv/inconsistency.csv")

    args = parser.parse_args()
    config = vars(args)

    device = config["device"].upper()
    incons_csv = config["infile"]

    # Set random seeds
    seed = 0
    random.seed(seed)
    tf.random.set_seed(seed)

    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

    # Create logic CPUs
    physical_devices = tf.config.list_physical_devices("CPU")
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         ])
    
    with open(incons_csv, 'r') as f:
        inconsistencies = f.readlines()

    model_id = -1
    for line in inconsistencies[1:]:
        flag = False
        if model_id == int(line.split(',')[0]) and input_id == int(line.split(',')[1]):
            flag = True
        model_id, input_id, setting, _ = line.strip().split(',')

        model_id = int(model_id)
        input_id = int(input_id)
        if model_id < NUM_MODELS // 2:
            continue

        # base
        base_setting = "0" + setting[1:]
        # check if layer_output exists:
        if os.path.exists(f"../results/analysis/output_{model_id}_{input_id}/output_{base_setting}.pk"):
            continue
        print(model_id, input_id, base_setting)
        layer_output(model_id, input_id, base_setting)

        # inconsistent
        if device == "CPU" and "CPU" in setting:
            print(model_id, input_id, setting)
            layer_output(model_id, input_id, setting)
        if device == "GPU" and "GPU" in setting:
            print(model_id, input_id, setting)
            layer_output(model_id, input_id, setting)

