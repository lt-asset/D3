from typing import *
import tensorflow as tf


def build_cmd(dist_config: dict) -> List[str]:
    cmd = ["python"]
    cmd.append("./worker.py")
    cmd.extend(["--model_type", f"{dist_config['model_type']}"])
    cmd.extend(["--model_idx", f"{dist_config['model_idx']}"])
    cmd.extend(["--input_idx", f"{dist_config['input_idx']}"])
    cmd.extend(["--device_type", f"{dist_config['device_type']}"])
    cmd.extend(["--num_device", f"{dist_config['num_device']}"])
    cmd.extend(["--strategy", f"{dist_config['strategy']}"])
    cmd.extend(["--frozen", f"{dist_config['frozen']}"])
    if 'extra' in dist_config and dist_config["extra"] is not None:
        cmd.extend(["--extra", f"{dist_config['extra']}"])
    return cmd


def generateStrategy(
        num_device: int, strategy: tf.distribute.Strategy, device_type: str = "GPU") -> tf.distribute.Strategy:
    assert (device_type in ["CPU", "GPU"])
    assert (num_device <= len(tf.config.list_logical_devices(device_type)))
    devices = []
    for i in range(num_device):
        devices.append("/{d}:{i}".format(d=device_type, i=i))
    # Create strategy with specific devices
    new_strategy = strategy(
        devices=devices
        # cross_device_ops=cross_device_ops.ReductionToOneDevice()
    )
    return new_strategy


def get_GPU_list(num_gpu: int) -> List[int]:
    gpu_lst: List[int] = list(range(1, min(4, num_gpu) + 1))
    if num_gpu >= 8:
        gpu_lst = gpu_lst + list([8])
    return gpu_lst