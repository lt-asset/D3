from test_utils import *
from test_DLRM import DLRMModelTest
from test_Sequential import SequentialModelTest
from constants import *
from typing import *
import random
import argparse
import traceback
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--model_idx", type=int, default=None)
    parser.add_argument("--input_idx", type=int, default=None)
    parser.add_argument("--device_type", type=str, default="CPU")
    parser.add_argument("--num_device", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="MirroredStrategy")
    parser.add_argument("--visible_gpus", type=str, default="-1")
    parser.add_argument("--frozen", type=int, default=0)
    parser.add_argument("--extra", type=str, default=None)

    args = parser.parse_args()
    config = vars(args)

    # Set random seeds
    seed = 0
    random.seed(seed)
    tf.random.set_seed(seed)

    # Setup environment
    if config["device_type"] == "CPU":
        # Disable GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Create logic CPUs
        cpu_devices = tf.config.list_physical_devices("CPU")
        tf.config.experimental.set_virtual_device_configuration(
            cpu_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             tf.config.experimental.VirtualDeviceConfiguration(),
             ])
    else:
        # Set visible GPUs
        gpu_str = config["visible_gpus"]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    # Generate strategy
    if config["strategy"] == "MirroredStrategy":
        strategy_type = tf.distribute.MirroredStrategy
    else:
        raise ("Strategy not recognized")
    strategy = generateStrategy(
        num_device=config["num_device"], strategy=strategy_type, device_type=config["device_type"])

    # Start the test
    dist_setting_str = str(config["num_device"]) + config["device_type"]
    extra = config["extra"]
    frozen = config["frozen"]

    try:
        if config["model_type"] == "DLRM":
            model_under_test = DLRMModelTest(
                model_idx=config["model_idx"],
                input_idx=config["input_idx"],
                strategy=strategy,
                dist_setting_str=dist_setting_str,
                model_extra=extra,
                frozen=frozen)
            print(model_under_test.model_idx,
                  model_under_test.input_idx,
                  dist_setting_str,
                  extra, 
                  frozen)
            model_under_test.test_DLRM_train()
        elif config["model_type"] == "Sequential":
            model_under_test = SequentialModelTest(
                model_idx=config["model_idx"],
                input_idx=config["input_idx"],
                strategy=strategy,
                dist_setting_str=dist_setting_str,
                model_extra=extra,
                frozen=frozen)
            print(model_under_test.model_idx,
                  model_under_test.input_idx,
                  dist_setting_str,
                  extra,
                  frozen)
            model_under_test.test_Sequential_train()
    except Exception as e:
        with open("../results/logs/error.log", "a+") as f:
            f.write(f"{config['model_idx']}-{config['input_idx']}-{dist_setting_str}-{extra}-f{frozen}\n")
            f.write(str(e))
            f.write("\n")
            f.write(traceback.format_exc())
            f.write("\n")

    exit(0)
