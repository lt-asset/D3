import os
import torch
import torch.nn as nn
from torchrec.distributed.test_utils.test_model import (
    TestSparseNNBase,
)

def generate_models(
    model_class: TestSparseNNBase,
    seed: int=0,
    model_save_path: str = "/data",
    regen: bool=False
) -> nn.Module:
    model = model_class(seed=seed)
    if os.path.exists(model_save_path) and not regen:
        # if the model state dict exist, pass
        model.load_state_dict(torch.load(model_save_path))
        print(model_save_path, "already exists")
    else:
        # if the model state dict does not exist, generate it and save it
        torch.save(model.state_dict(), model_save_path)
    return model

