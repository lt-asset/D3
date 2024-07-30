from functools import partial
import os
from dlrm.dlrm import DLRM
from test_utils import *
from constants import *
from keras import optimizers
from tensorflow import Tensor
from typing import *
import tensorflow as tf
import pickle
import json
import keras
import tensorflow_model_optimization as tfmot
from convert_to_quantized_with_datatype import apply_quantization
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

class DLRMModelTest():
    """
    Class for testing DLRM models
    """

    def __init__(self, model_idx: int, input_idx: int, strategy: tf.distribute.Strategy, dist_setting_str: str, model_extra: str, frozen: int):
        self.model_idx = model_idx
        self.input_idx = input_idx
        self.strategy = strategy
        self.dist_setting_str = dist_setting_str
        self.model_extra = model_extra
        self.frozen = frozen
        self.model_dir = f"../data/models/model_{model_idx}/"
        self.input_dir = f"../data/inputs/input_{model_idx}/"
        self.result_dir = f"../results/outputs/output_{model_idx}/"
        with open(self.model_dir + "model_spec.json", 'r') as f:
            self.model_specs = json.load(f)

    def _DLRM_train_nondist(self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset) -> Tensor:
        # Load model

        model: DLRM = DLRM(
            num_embed=self.model_specs["num_embed"],
            embed_dim=self.model_specs["embed_dim"],
            embed_vocab_size=self.model_specs["embed_vocab_size"],
            ln_bot=self.model_specs["ln_bot"],
            ln_top=self.model_specs["ln_top"])
        model.load_weights(self.model_dir + "weights/")
        optimizer = optimizers.Adam()

        for data in test_dataset:
            _ = model.inference(data["dense_features"], data["sparse_features"])

        if self.model_extra in ["quantize_8_8"]:
            num_bits_weight = int(self.model_extra.split("_")[1])
            num_bits_activation = int(self.model_extra.split("_")[2])
            with tfmot.quantization.keras.quantize_scope():
                partial_apply_quantization = partial(apply_quantization, num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation)

                annotated_model = tf.keras.models.clone_model(
                    model._mlp_bot,
                    clone_function=partial_apply_quantization)
                model._mlp_bot = tfmot.quantization.keras.quantize_apply(annotated_model,
                    tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation))

                annotated_model = tf.keras.models.clone_model(
                    model._mlp_top,
                    clone_function=partial_apply_quantization)
                model._mlp_top = tfmot.quantization.keras.quantize_apply(annotated_model,
                    tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation))
                
        if self.frozen > 0 and os.path.exists(self.model_dir + "frozen_order.txt"):
            with open(self.model_dir + "frozen_order.txt", 'r') as f:
                frozen_order = eval(f.read())
            for i in range(min(len(frozen_order), self.frozen)):
                model.layers[frozen_order[i]].trainable = False

        def train_step(dense_features: Tensor, sparse_features: Tensor, label: Tensor):
            with tf.GradientTape() as tape:
                loss_value = model.get_myloss(
                    dense_features, sparse_features, label)
            gradients = tape.gradient(
                loss_value, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            return loss_value

        # Train for 1 step
        for data in train_dataset:
            loss: Tensor = train_step(
                data["dense_features"], data["sparse_features"], data["label"])

        # Test
        for data in test_dataset:
            pred: Tensor = model.inference(
                data["dense_features"], data["sparse_features"])

        return pred

    def _DLRM_train_dist(self, strategy: tf.distribute.Strategy, train_dataset: tf.data.Dataset,
                         test_dataset: tf.data.Dataset) -> Tensor:
        train_dataset_dist = strategy.experimental_distribute_dataset(
            train_dataset)
        test_dataset_dist = strategy.experimental_distribute_dataset(
            test_dataset)
        
        # Test
        @tf.function
        def distributed_inference(dist_inputs: tf.data.Dataset):
            per_replica_prediction = strategy.run(model_dist.inference, args=(
                dist_inputs["dense_features"], dist_inputs["sparse_features"]))
            return per_replica_prediction
        
        # Load model
        with strategy.scope():
            model_dist: DLRM = DLRM(
                num_embed=self.model_specs["num_embed"],
                embed_dim=self.model_specs["embed_dim"],
                embed_vocab_size=self.model_specs["embed_vocab_size"],
                ln_bot=self.model_specs["ln_bot"],
                ln_top=self.model_specs["ln_top"])

            model_dist.load_weights(self.model_dir + "weights/")
            optimizer = optimizers.SGD(learning_rate=10)

            for dist_inputs in test_dataset_dist:
                _ = distributed_inference(dist_inputs)

            if self.model_extra in ["quantize_8_8"]:
                num_bits_weight = int(self.model_extra.split("_")[1])
                num_bits_activation = int(self.model_extra.split("_")[2])
                with tfmot.quantization.keras.quantize_scope():
                    partial_apply_quantization = partial(apply_quantization, num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation)

                    annotated_model = tf.keras.models.clone_model(
                        model_dist._mlp_bot,
                        clone_function=partial_apply_quantization)
                    model_dist._mlp_bot = tfmot.quantization.keras.quantize_apply(annotated_model,
                        tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation))

                    annotated_model = tf.keras.models.clone_model(
                        model_dist._mlp_top,
                        clone_function=partial_apply_quantization)
                    model_dist._mlp_top = tfmot.quantization.keras.quantize_apply(annotated_model,
                        tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(num_bits_weight=num_bits_weight, num_bits_activation=num_bits_activation))
                    
            if self.frozen > 0 and os.path.exists(self.model_dir + "frozen_order.txt"):
                with open(self.model_dir + "frozen_order.txt", 'r') as f:
                    frozen_order = eval(f.read())
                for i in range(min(len(frozen_order), self.frozen)):
                    if frozen_order[i] == 0:
                        model_dist._mlp_bot.trainable = False
                    elif frozen_order[i] == 1:
                        model_dist._mlp_top.trainable = False

        def train_step(dense_features: Tensor, sparse_features: Tensor, label: Tensor):
            with tf.GradientTape() as tape:
                loss_value = model_dist.get_myloss_dist(
                    dense_features, sparse_features, label, BATCH_SIZE)
            gradients = tape.gradient(
                loss_value, model_dist.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model_dist.trainable_variables))
            return loss_value

        @tf.function
        def distributed_train_step(dist_inputs):
            per_replica_losses = strategy.run(train_step, args=(
                dist_inputs["dense_features"], dist_inputs["sparse_features"], dist_inputs["label"]))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        # Train for 1 step
        for dist_inputs in train_dataset_dist:
            loss: Tensor = distributed_train_step(dist_inputs)

        for dist_inputs in test_dataset_dist:
            pred = distributed_inference(dist_inputs)

        if strategy.num_replicas_in_sync > 1:
            pred = tf.concat(pred.values, axis=0)

        return pred

    def test_DLRM_train(self):
        # Sharding policy options for datasets
        options = tf.data.Options()
        options.deterministic = True
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # Load test data
        with open(self.input_dir + f"test_{self.input_idx}.pk", 'rb') as f:
            (test_dense, test_sparse, test_label) = pickle.load(f)
        test_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices({
            'dense_features': test_dense,
            'sparse_features': test_sparse,
            'label': test_label
        }).batch(BATCH_SIZE).with_options(options)

        # Load training data
        with open(self.input_dir + f"train_{self.input_idx}.pk", 'rb') as f:
            (train_dense, train_sparse, train_label) = pickle.load(f)
        train_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices({
            'dense_features': train_dense,
            'sparse_features': train_sparse,
            'label': train_label
        }).batch(BATCH_SIZE).with_options(options)

        num_devices = int(self.dist_setting_str[0])
        # One step train and test
        if num_devices == 0:
            pred = self._DLRM_train_nondist(train_dataset, test_dataset)
        else:
            pred = self._DLRM_train_dist(self.strategy, train_dataset, test_dataset)
        
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

