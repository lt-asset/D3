from keras import optimizers, losses
from test_utils import *
from constants import *
from tensorflow import Tensor
import tensorflow as tf
import pickle
import keras
import random
import os
import tensorflow_model_optimization as tfmot

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()


class SequentialModelTest():
    def __init__(
            self, model_idx: int, input_idx: int, strategy: tf.distribute.Strategy, dist_setting_str: str,
            model_extra: str,
            frozen: int):
        self.model_idx = model_idx
        self.input_idx = input_idx
        self.strategy = strategy
        self.dist_setting_str = dist_setting_str
        self.model_extra = model_extra
        self.frozen = frozen
        self.model_dir = f"../data/models/model_{model_idx}/"
        self.input_dir = f"../data/inputs/input_{model_idx}/"
        self.result_dir = f"../results/outputs/output_{model_idx}/"

        os.makedirs(self.result_dir, exist_ok=True)

    def _Sequential_train_nondist(
            self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset) -> Tensor:
        # Load model
        if self.model_extra in ["quantize_8_8"]:
            num_bits_weight = int(self.model_extra.split("_")[1])
            num_bits_activation = int(self.model_extra.split("_")[2])
            with tfmot.quantization.keras.quantize_scope():
                model = keras.models.load_model(
                    self.model_dir + f"model_quantized_{num_bits_weight}_{num_bits_activation}.h5")
        else:
            model = keras.models.load_model(
                self.model_dir + "model.h5")
            
        if self.frozen > 0 and os.path.exists(self.model_dir + "frozen_order.txt"):
            with open(self.model_dir + "frozen_order.txt", 'r') as f:
                frozen_order = eval(f.read())
            for i in range(min(len(frozen_order), self.frozen)):
                model.layers[frozen_order[i]].trainable = False

        optimizer = optimizers.SGD(learning_rate=10)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.AUTO)

        model.compile(optimizer=optimizer, loss=loss)

        # Train for 1 step
        loss: Tensor = model.fit(
            train_dataset[0], train_dataset[1], verbose=0, shuffle=False, batch_size=2400)

        # pred: Tensor = model_dist.predict(test_dataset[0], verbose=0)
        pred: Tensor = model(test_dataset[0])

        return pred

    def _Sequential_train_dist(self, strategy: tf.distribute.Strategy, train_dataset: tf.data.Dataset,
                               test_dataset: tf.data.Dataset) -> Tensor:
        # Load model
        with strategy.scope():
            # Load model
            if self.model_extra in ["quantize_8_8"]:
                num_bits_weight = int(self.model_extra.split("_")[1])
                num_bits_activation = int(self.model_extra.split("_")[2])
                with tfmot.quantization.keras.quantize_scope():
                    model = keras.models.load_model(
                        self.model_dir + f"model_quantized_{num_bits_weight}_{num_bits_activation}.h5")
            else:
                model = keras.models.load_model(
                    self.model_dir + "model.h5")
                
            if self.frozen > 0 and os.path.exists(self.model_dir + "frozen_order.txt"):
                with open(self.model_dir + "frozen_order.txt", 'r') as f:
                    frozen_order = eval(f.read())
                for i in range(min(len(frozen_order), self.frozen)):
                    model.layers[frozen_order[i]].trainable = False

            optimizer = optimizers.SGD(learning_rate=10)
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.AUTO)

            model.compile(optimizer=optimizer, loss=loss)

        # Train for 1 step
        loss: Tensor = model.fit(
            train_dataset[0], train_dataset[1], verbose=0, shuffle=False, batch_size=2400)

        # pred: Tensor = model_dist.predict(test_dataset[0], verbose=0)
        pred: Tensor = model(test_dataset[0])

        return pred

    def test_Sequential_train(self):
        # Load test data
        with open(self.input_dir + f"test_{self.input_idx}.pk", 'rb') as f:
            (test_input, test_label) = pickle.load(f)

        # Load training data
        with open(self.input_dir + f"train_{self.input_idx}.pk", 'rb') as f:
            (train_input, train_label) = pickle.load(f)
        
        num_devices = int(self.dist_setting_str[0])

        # One step train and test
        if num_devices == 0:
            pred = self._Sequential_train_nondist(
                train_dataset=(train_input, train_label),
                test_dataset=(test_input, test_label))
        else:
            pred = self._Sequential_train_dist(
                strategy=self.strategy,
                train_dataset=(train_input, train_label),
                test_dataset=(test_input, test_label))

        suffix = ""
        if self.model_extra in ["quantize_8_8"]:
            suffix += f"_{self.model_extra}"
        if self.frozen > 0:
            frozen_suffix = f"_f{self.frozen}"
        else:
            frozen_suffix = ""
        # Save output
        with open(self.result_dir + f"output_{self.input_idx}_{self.dist_setting_str}{suffix}{frozen_suffix}.pk", 'wb') as f:
            pickle.dump(pred, f)
