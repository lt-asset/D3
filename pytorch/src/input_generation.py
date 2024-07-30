from typing import List, Optional, Tuple
import os
import random
import numpy as np
import pickle

import torch
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.test_utils.test_model import ModelInput

def generate_single_input(
    tables: List[EmbeddingTableConfig],
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    world_size: int = 1,
    batch_size: int = 2400,
    num_float_features: int = 16,
) -> Tuple[ModelInput, List[ModelInput]]:
    return ModelInput.generate(
        batch_size=batch_size,
        world_size=world_size,
        num_float_features=num_float_features,
        tables=tables,
        weighted_tables=weighted_tables or [],
    )

def generate_inputs(
    model,
    input_seed,
    world_size: int = 1,
    batch_size: int = 2400,
    input_save_path: str = "/data",
    regen: bool=False
):
    torch.manual_seed(input_seed)
    random.seed(input_seed)
    np.random.seed(input_seed)

    if os.path.exists(input_save_path) and not regen:
        # if the input exists, load it
        print(input_save_path, "already exists")
        with open(input_save_path, "rb") as f:
            inputs = pickle.load(f)
    else:
        inputs = [
            generate_single_input(
                tables=model.tables,
                weighted_tables=model.weighted_tables,
                world_size=world_size,
                batch_size=batch_size,
                num_float_features=model.in_features,
            ),
            generate_single_input(
                tables=model.tables,
                weighted_tables=model.weighted_tables,
                world_size=world_size,
                batch_size=batch_size,
                num_float_features=model.in_features,
            )
        ]
        with open(input_save_path, "wb") as f:
            pickle.dump(inputs, f)

    return inputs

def generate_sequential_input(
    input_seed,
    batch_size: int = 2400,
    input_save_path: str = "/data",
    regen: bool=False
):
    torch.manual_seed(input_seed)
    random.seed(input_seed)
    np.random.seed(input_seed)

    if os.path.exists(input_save_path) and not regen:
        # if the input exists, load it
        print(input_save_path, "already exists")
        with open(input_save_path, "rb") as f:
            inputs = pickle.load(f)
    else:
        train_input = np.random.uniform(size=(batch_size, 32, 32, 3))
        train_label = np.zeros((batch_size, 10))
        for i in range(batch_size):
            onehot_index = random.randint(0, 9)
            train_label[i, onehot_index] = 1

        test_input = np.random.uniform(size=(batch_size, 32, 32, 3))
        test_label = np.zeros((batch_size, 10))
        for i in range(batch_size):
            onehot_index = random.randint(0, 9)
            test_label[i, onehot_index] = 1
        np_input = ((train_input, train_label), (test_input, test_label))

        with open(input_save_path, 'wb') as f:
            pickle.dump(np_input, f)

