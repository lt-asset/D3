import random
import json
from model_generation import *
from constants import *


if __name__ == "__main__":
    random.seed(0)
    for i in range(NUM_MODELS//4):
        seed = random.randint(0, 1e8)
        print(f"Generating DLRM model {i} with seed {seed}")
        generate_DLRM_model(idx=i, seed=seed)
        print(f"Generating DLRM model {i + NUM_MODELS//4} with seed {seed}")
        generate_DLRM_model(idx=(i + NUM_MODELS//4), seed=seed)
    for i in range(NUM_MODELS//2, NUM_MODELS//2 + NUM_MODELS//4):
        seed = random.randint(0, 1e8)
        print(f"Generating Sequential model {i} with seed {seed}")
        generate_Sequential_model(idx=i, seed=seed)
        print(f"Generating Sequential model {i + NUM_MODELS//4} with seed {seed}")
        generate_Sequential_model(idx=i + NUM_MODELS//4, seed=seed)

