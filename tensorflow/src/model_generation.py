import os
import json
import pickle
import random
import subprocess
import keras
import tensorflow as tf
from typing import *
from constants import *
from dlrm.dlrm import DLRM
from input_generation import *


def generate_DLRM_model(
        idx: int, seed: int, num_embed_bag: int = None, embed_vocab_size: List[int] = None, embed_dim: int = None,
        bot_out_feature: int = None, top_out_feature: int = None, size_dense: int = None):

    def save_model(model: DLRM, test_dataset: tf.data.Dataset, path: str, seed: int):
        # Make one inference to get input size
        for d in test_dataset:
            model.inference(d["dense_features"], d["sparse_features"])
            break

        model_specs = model.get_params()
        model_specs["seed"] = seed
        with open(f"{path}/model_spec.json", 'w') as f:
            json.dump(model_specs, f, indent=2)
        model.save_weights(f"{path}/weights/")
        # keras.models.save_model(model, path + "model.tf", save_format="tf")

    cur_dir = os.getcwd()
    model_path = f"{cur_dir}/../data/models/model_{idx}"
    input_path = f"{cur_dir}/../data/inputs/input_{idx}"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # Generate model
    # Number of embedding bags
    if num_embed_bag is None:
        num_embed_bag = random.randint(1, 5)

    # Embedding vocab size
    if embed_vocab_size is None:
        embed_vocab_size = []
        for _ in range(num_embed_bag):
            embed_vocab_size.append(random.randint(1, 1000))
    else:
        assert (len(embed_vocab_size) == num_embed_bag)

    # Embedding dimension
    if embed_dim is None:
        embed_dim = 4 * random.randint(1, 250)

    # Linear layers
    if bot_out_feature is None:
        bot_out_feature = random.randint(1, 1000)
    if top_out_feature is None:
        top_out_feature = random.randint(1, 1000)

    model = DLRM(
        num_embed=num_embed_bag,
        embed_dim=embed_dim,
        embed_vocab_size=embed_vocab_size,
        ln_bot=bot_out_feature,
        ln_top=top_out_feature,
    )

    # Generate input
    if size_dense is None:
        size_dense = random.randint(1, 1000)

    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
    with open(f"{input_path}/SEED", 'w') as f:
        f.write(str(seed))
    generate_DLRM_input(input_path, size_dense, embed_vocab_size)

    with open(f"{input_path}/test_0.pk", 'rb') as f:
        (test_dense, test_sparse, test_label) = pickle.load(f)

    test_dataset = tf.data.Dataset.from_tensor_slices({
        'dense_features': test_dense,
        'sparse_features': test_sparse,
        'label': test_label
    }).batch(BATCH_SIZE)

    # Save model
    save_model(model=model, test_dataset=test_dataset,
               path=model_path, seed=seed)


def generate_Sequential_model(idx: int, seed: int):
    cur_dir = os.getcwd()
    input_path = f"{cur_dir}/../data/inputs/input_{idx}/"
    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
    with open(f"{cur_dir}/../data/inputs/input_{idx}/SEED", 'w') as f:
        f.write(str(seed))
    random.seed(seed)
    tf.random.set_seed(seed)
    generate_Sequential_input(input_path)
