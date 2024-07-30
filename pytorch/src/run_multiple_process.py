from multiprocessing import Process
import multiprocessing
import os
from timeit import default_timer as timer
import time
import traceback
import subprocess
from config import distributed_settings, DistributedConfig
from constants import BATCH_SIZE, NUM_MODELS, NUM_TRAINING_DATASETS

def build_command(setting:DistributedConfig, model_state_dict_path, input_path, seed, input_seed):
    quantization = setting.quantization
    sharder_type = setting.sharder_type
    sharding_type = setting.sharding_type
    kernel_type = setting.kernel_type
    backend = setting.backend
    world_size = setting.world_size

    command = "python -m worker "
    if quantization:
        command += " --quantization"
    command += " --sharder_type " + sharder_type
    command += " --sharding_type " + sharding_type
    command += " --kernel_type " + kernel_type
    command += " --backend " + backend
    command += " --world_size " + str(world_size)
    command += " --model_state_dict_path " + model_state_dict_path
    command += " --input_path " + input_path
    command += " --seed " + str(seed)
    command += " --input_seed " + str(input_seed)
    command += " --batch_size " + str(BATCH_SIZE)
    return command

def execute_one_single_run(task_queue, gpu):
    try:
        # lock_acquired = False
        while True:
            # get the run
            try:
                command = task_queue.get(block=False)
            except Exception:
                return

            print('RUNNING: ' + command)
            if gpu is not None:
                command = "CUDA_VISIBLE_DEVICES=" + gpu + " " + command

            start_time = timer()
            status = subprocess.call(command, shell=True)
            end_time = timer()
            test_time = end_time - start_time

    except Exception:
        print(command)
        print(traceback.format_exc())


if __name__ == "__main__":
    start = time.time()
    print("start time: ", start)

    num_models = NUM_MODELS
    num_inputs = NUM_TRAINING_DATASETS

    command_list = [[] for _ in range(5)]  # [0] is cpu, [1] is one gpu, [2] is two gpus, [3] is three or four gpus, [4] is eight gpus
    procs = []
    for i in range(num_models):
        seed = i
        if i < int(num_models/2):
            model_state_dict_path = os.path.join("/data", "models", "model_{}".format(seed))
        else:
            model_state_dict_path = os.path.join("/data", "models", "model_{}.onnx".format(seed))
        
        for j in range(num_inputs):
            input_index = j
            input_path = os.path.join("/data", "inputs", "input_{}".format(seed), "input_{}_{}".format(seed, input_index))
            for setting in distributed_settings:
                if setting.quantization:
                    continue
                command = build_command(setting, model_state_dict_path, input_path, seed, input_index)
                if setting.backend == "gloo":
                    command_list[0].append(command)
                else:
                    if setting.world_size == 1:
                        command_list[1].append(command)
                    elif setting.world_size == 2:
                        command_list[2].append(command)
                    elif setting.world_size == 3 or setting.world_size == 4:
                        command_list[3].append(command)
                    else:  # world size == 8
                        command_list[4].append(command)
    
    # for cpu processes:
    queue = multiprocessing.Queue()
    for command in command_list[0]:
        queue.put(command)
    NUM_PROCESSES = 12
    processes = []
    for p_index in range(NUM_PROCESSES):
        # start each processes
        process = Process(target=execute_one_single_run,
                          args=(queue, None))
        process.start()
        processes.append(process)
    # wait for all processes to finish
    for process in processes:
        process.join()

    # for rank 1 gpu processes:
    queue = multiprocessing.Queue()
    for command in command_list[1]:
        queue.put(command)
    processes = []
    gpu_list = ["0", "1", "2", "3", "4", "5", "6", "7"]
    for p_index in range(len(gpu_list)):
        # start each processes
        process = Process(target=execute_one_single_run,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # wait for all processes to finish
    for process in processes:
        process.join()

    # for rank 2 gpu processes:
    queue = multiprocessing.Queue()
    for command in command_list[2]:
        queue.put(command)
    processes = []
    gpu_list = ["0,1", "2,3", "4,5", "6,7"]
    for p_index in range(len(gpu_list)):
        # start each processes
        process = Process(target=execute_one_single_run,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # wait for all processes to finish
    for process in processes:
        process.join()
    
    # for rank 4 gpu processes:
    queue = multiprocessing.Queue()
    for command in command_list[3]:
        queue.put(command)
    processes = []
    gpu_list = ["0,1,2,3", "4,5,6,7"]
    for p_index in range(len(gpu_list)):
        # start each processes
        process = Process(target=execute_one_single_run,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # wait for all processes to finish
    for process in processes:
        process.join()
    
    # for rank 8 gpu processes:
    queue = multiprocessing.Queue()
    for command in command_list[4]:
        queue.put(command)
    processes = []
    gpu_list = ["0,1,2,3,4,5,6,7"]
    for p_index in range(len(gpu_list)):
        # start each processes
        process = Process(target=execute_one_single_run,
                          args=(queue, gpu_list[p_index]))
        process.start()
        processes.append(process)
    # wait for all processes to finish
    for process in processes:
        process.join()
    
    end = time.time()
    print("end time: ", end)
    print("total time: ", end - start)