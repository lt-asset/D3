import os
import tensorflow as tf
import tf2onnx
import gc

from constants import NUM_MODELS

def transform_onnx(model_path: str, onnx_path: str):
    if os.path.exists(onnx_path):
        return
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # 5G memory limitation
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    model = tf.keras.models.load_model(model_path)
    if tf.__version__.split(".")[0] == 1:
        input_shape = model.layers[0].input_shape
    else:
        input_shape = model.layers[0].input_shape[0]
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    _, _ = tf2onnx.convert.from_keras(model, input_signature=spec, \
        opset=15, output_path=onnx_path)
    del model
    del _

if __name__ == "__main__":
    tensorflow_model_dir = "/tensorflow/data/models"
    pytorch_model_dir = "/data/models"

    index = int(NUM_MODELS/2)

    while index < NUM_MODELS:
        model_path = os.path.join(tensorflow_model_dir, "model_{}".format(index), "model.h5")
        if not os.path.exists(model_path):
            print("{} not exists".format(model_path))
            index += 1
            continue
        onnx_path = os.path.join(pytorch_model_dir, "model_{}.onnx".format(index))
        transform_onnx(model_path, onnx_path)
        index += 1

    gc.collect()