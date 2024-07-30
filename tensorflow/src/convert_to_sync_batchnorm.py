import os
import logging
import keras
import tensorflow as tf
from constants import *
from keras import Model, layers, Input

tf.get_logger().setLevel(logging.ERROR)

def convert_to_sync_batch_norm(old_model: Model, input_layer: Input):
    old_layer_names = [layer.name for layer in old_model.layers]
    new_xs = [input_layer]
    for old_layer in old_model.layers[1:]:
        if isinstance(old_layer.input, list):
            input_x = [new_xs[old_layer_names.index(
                l.name.split("/")[0])] for l in old_layer.input]
        else:
            input_x = new_xs[old_layer_names.index(
                old_layer.input.name.split("/")[0])]
        if isinstance(old_layer, layers.BatchNormalization):
            old_layer = tf.keras.layers.experimental.SyncBatchNormalization.from_config(
                old_layer.get_config()
            )
        x = old_layer(input_x)
        new_xs.append(x)

    new_model = Model(new_xs[0], new_xs[-1])
    for old_layer, new_layer in zip(old_model.layers, new_model.layers):
        new_layer.set_weights(old_layer.get_weights())

    return new_model


def get_models_with_batchnorm(model_dir):
    # Get the list of models that contain batchnorm layers
    batchnorm_lst = []
    for m in range(NUM_MODELS//2, NUM_MODELS):
        with open(f"{model_dir}/model_{m}/model.json", 'r') as f:
            model_spec = f.read()
        if "batch_normalization" in model_spec:
            batchnorm_lst.append(m)
    return batchnorm_lst


if __name__ == "__main__":
    print("Converting BatchNormalization to SyncBatchNormalization")
    cur_dir = os.getcwd()
    model_dir = f"{cur_dir}/../data/models"
    models_with_batchnorm = get_models_with_batchnorm(model_dir)
    print("Models with BatchNormalization:", models_with_batchnorm)
    for m in models_with_batchnorm:
        try:
            original_model: Model = keras.models.load_model(f"{model_dir}/model_{m}/model.h5")
            converted_model = convert_to_sync_batch_norm(original_model, Input(shape=(32, 32, 3)))
            keras.models.save_model(converted_model, f"{model_dir}/model_{m}/model_syncbatch.h5")
            print(f"Model {m} converted")
        except Exception as e:
            print(str(e))
            print("Conversion failed")
