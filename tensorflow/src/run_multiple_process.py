import multiprocessing
import time
import subprocess
from subprocess import CalledProcessError
from queue import Empty
from multiprocessing import Process
import os
import sys
import itertools
from typing import *
from constants import *
from test_utils import build_cmd


def print_log(logname: str, content: str):
    if not LOG:
        return
    with open(logname, "a") as f:
        f.write(content + '\n')


def execute_single_worker(task_queue: multiprocessing.Queue, gpu: str):
    while True:
        try:
            command: List[str] = task_queue.get(block=False)
        except Empty:
            return

        if gpu:
            command.extend(["--visible_gpus", f"{gpu}"])
        print("Running command:", ' '.join(command))

        try:
            subprocess.check_call(command, timeout=120.0)

        except CalledProcessError:
            print_log(logname, f"[FAILED] {command}")


if __name__ == "__main__":
    input_idx_lst = list(range(NUM_TRAINING_DATASETS))
    device_type_lst = ["GPU", "CPU"]
    # num_device_lst = [0, 1, 2, 3, 4, 8]
    num_device_lst = [0]
    strategy_lst = ["MirroredStrategy"]
    extra_lst = [None, "quantize_8_8"]
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logname = f"../results/logs/{curr_time}.log"
    LOG = True

    
    cmd_lst = [[] for _ in range(5)]  # Contains 5 lists, [0]=cpu, [1]=1gpu, [2]=2gpu, [3]=3gpu or 4gpu, [4]=8gpu

    for model_idx in range(NUM_MODELS):
        model_type = "DLRM" if model_idx < NUM_MODELS // 2 else "Sequential"
        # Result dir
        result_path = f"../results/outputs/output_{model_idx}"
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

        for (input_idx, device_type, num_device, strategy, extra) in itertools.product(input_idx_lst, device_type_lst, num_device_lst, strategy_lst, extra_lst):
            if os.path.exists(f"../data/models/model_{model_idx}/frozen_order.txt"):
                frozen = 1
            else:
                frozen = 0
            config = {
                "model_type": model_type,
                "model_idx": model_idx,
                "input_idx": input_idx,
                "device_type": device_type,
                "num_device": num_device,
                "strategy": strategy,
                "extra": extra,
                "frozen": frozen
            }
            cmd = build_cmd(config)
            if device_type == "CPU":
                cmd_lst[0].append(cmd)
            else:  # device_type == GPU
                if num_device == 1 or num_device == 0:
                    cmd_lst[1].append(cmd)
                elif num_device == 2:
                    cmd_lst[2].append(cmd)
                elif num_device == 3 or num_device == 4:
                    cmd_lst[3].append(cmd)
                elif num_device == 8:
                    cmd_lst[4].append(cmd)

    total_time = time.time()

    # For cpu processes:
    queue = multiprocessing.Queue()
    for cmd in cmd_lst[0]:
        queue.put(cmd)
    # Start each processes
    start_time = time.time()
    processes = []
    NUM_CPU_PROCESSES = 12
    for p_index in range(NUM_CPU_PROCESSES):
        process = Process(target=execute_single_worker,
                          args=(queue, None))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()
    time_spent = time.time() - start_time
    print_log(logname, f"Time spent in CPU processes: {time_spent}")

    # For GPU processes
    # num_device == 1
    queue = multiprocessing.Queue()
    for cmd in cmd_lst[1]:
        queue.put(cmd)
    # Start each processes
    start_time = time.time()
    processes = []
    gpu_list = ["0", "1", "2", "3", "4", "5", "6", "7"]
    for p_index in range(len(gpu_list)):
        process = Process(target=execute_single_worker,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()
    time_spent = time.time() - start_time
    print_log(logname, f"Time spent in 1GPU processes: {time_spent}")

    # # num_device == 2
    queue = multiprocessing.Queue()
    for cmd in cmd_lst[2]:
        queue.put(cmd)
    # Start each processes
    start_time = time.time()
    processes = []
    gpu_list = ["0,1", "2,3", "4,5", "6,7"]
    for p_index in range(len(gpu_list)):
        process = Process(target=execute_single_worker,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()
    time_spent = time.time() - start_time
    print_log(logname, f"Time spent in 2GPU processes: {time_spent}")

    # num_device == 3,4
    queue = multiprocessing.Queue()
    for cmd in cmd_lst[3]:
        queue.put(cmd)
    # Start each processes
    start_time = time.time()
    processes = []
    gpu_list = ["0,1,2,3", "4,5,6,7"]
    for p_index in range(len(gpu_list)):
        process = Process(target=execute_single_worker,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()
    time_spent = time.time() - start_time
    print_log(logname, f"Time spent in 3, 4GPU processes: {time_spent}")

    # num_device == 8
    queue = multiprocessing.Queue()
    for cmd in cmd_lst[4]:
        queue.put(cmd)
    # Start each processes
    start_time = time.time()
    processes = []
    gpu_list = ["0,1,2,3,4,5,6,7"]
    for p_index in range(len(gpu_list)):
        process = Process(target=execute_single_worker,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()
    time_spent = time.time() - start_time
    print_log(logname, f"Time spent in 8GPU processes: {time_spent}")

    time_spent = time.time() - total_time
    print_log(logname, f"Time spent: {time_spent}")
