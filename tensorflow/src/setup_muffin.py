import os
import json
import argparse
from constants import *

if __name__ == "__main__":
    cur_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="seq")

    args = parser.parse_args()
    config = vars(args)

    muffin_config = {
        "debug_mode": 1,
        "dataset_name": "cifar10",
        "case_num": NUM_MODELS//8,
        "generate_mode": config["mode"],
        "data_dir": "data",
        "timeout": 60,
        "use_heuristic": 1
    }

    with open(f"{cur_dir}/muffin/testing_config.json", 'w') as f:
        json.dump(muffin_config, f, indent=2)
