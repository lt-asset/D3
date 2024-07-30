import os
import keras
import logging
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras import Model, layers
from constants import *
from convert_to_sync_batchnorm import get_models_with_batchnorm
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_registry as n_bit_registry
from functools import partial
import itertools

tf.get_logger().setLevel(logging.ERROR)

class _QuantizeInfo(object):
  """QuantizeInfo."""

  def __init__(self,
               layer_type,
               weight_attrs,
               activation_attrs,
               quantize_output=False,
               num_bits_weight=4,
               num_bits_activation=16):
    """QuantizeInfo.

    Args:
      layer_type: Type of keras layer.
      weight_attrs: List of quantizable weight attributes of layer.
      activation_attrs: List of quantizable activation attributes of layer.
      quantize_output: Bool. Should we quantize the output of the layer.
      num_bits_weight: Int. The number of bits for the weight. Default to 8.
      num_bits_activation: Int. The number of bits for the activation.
                           Default to 8.
    """
    self.layer_type = layer_type
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output
    self.num_bits_weight = num_bits_weight
    self.num_bits_activation = num_bits_activation

quantize_info = {
    layers.Conv1D: _QuantizeInfo(layers.Conv1D, ["kernel"], ["activation"]),
    layers.Conv2D: _QuantizeInfo(layers.Conv2D, ["kernel"], ["activation"]),
    layers.Conv3D: _QuantizeInfo(layers.Conv3D, ["kernel"], ["activation"]),
    layers.DepthwiseConv1D: _QuantizeInfo(layers.DepthwiseConv1D, ["depthwise_kernel"], ["activation"]),
    layers.DepthwiseConv2D: _QuantizeInfo(layers.DepthwiseConv2D, ["depthwise_kernel"], ["activation"]),
    layers.Dense: _QuantizeInfo(layers.Dense, ["kernel"], ["activation"]),
    layers.ReLU: _QuantizeInfo(layers.ReLU, [], [], True),
    layers.Concatenate: _QuantizeInfo(layers.Concatenate, [], [], True),
    layers.Add: _QuantizeInfo(layers.Add, [], [], True)
}

def build_quantize_config(quantize_info, num_bits_weight=4, num_bits_activation=16):
    """Build quantize config."""
    return n_bit_registry.DefaultNBitQuantizeConfig(quantize_info.weight_attrs,
                                     quantize_info.activation_attrs,
                                     quantize_info.quantize_output,
                                     num_bits_weight,
                                     num_bits_activation)

# List of layers that can be quantized
quantizable_layers = [
    layers.Conv1D,
    layers.Conv2D,
    layers.Conv3D,
    layers.DepthwiseConv1D,
    layers.DepthwiseConv2D,
    layers.Dense,
    layers.ReLU,
    layers.Concatenate,
    layers.Add,
]


def apply_quantization(layer, num_bits_weight, num_bits_activation):
  if type(layer) in quantizable_layers:
    return tfmot.quantization.keras.quantize_annotate_layer(layer, build_quantize_config(quantize_info[type(layer)], num_bits_weight, num_bits_activation))
  return layer

supported_bit_weight = [
   8
]

supported_bit_activation = [
   8
]


if __name__ == "__main__":
    print("Generating quantized models")
    cur_dir = os.getcwd()
    model_dir = f"{cur_dir}/../data/models"
    # models_with_batchnorm = get_models_with_batchnorm(model_dir)
    models_with_batchnorm = []
    for m in range(NUM_MODELS//2, NUM_MODELS):
        for num_bits_weight, num_bits_activation in itertools.product(supported_bit_weight, supported_bit_activation):
            try:
                original_model: Model = keras.models.load_model(f"{model_dir}/model_{m}/model.h5")

                with tfmot.quantization.keras.quantize_scope():
                    partial_apply_quantization = partial(apply_quantization, num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation)
                    annotated_model = tf.keras.models.clone_model(
                        original_model,
                        clone_function=partial_apply_quantization)
                    converted_model = tfmot.quantization.keras.quantize_apply(annotated_model,
                        tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation))
                keras.models.save_model(converted_model, f"{model_dir}/model_{m}/model_quantized_{num_bits_weight}_{num_bits_activation}.h5")
                print(f"Model {m} converted")
            except Exception as e:
                print(str(e))
                print(f"Model {m} conversion failed")
