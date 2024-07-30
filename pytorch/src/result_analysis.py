import pickle
import os
import sys
import traceback
import torch
import numpy as np
import csv
from config import DistributedConfig, distributed_settings, QuantizationConfig, quantization_settings
from constants import BATCH_SIZE, NUM_MODELS, NUM_TRAINING_DATASETS
from test_utils import load_intermediate_result, load_pred_result, load_intermediate_result_sequential_lzma, load_pred_result_lzma, compare_intermidiate_results

def normalize_pred(pred):
    pred = torch.nn.functional.softmax(pred, -1)
    return pred

def construct_unit_test_str(setting: DistributedConfig):
    quantization = setting.quantization
    sharder_type = setting.sharder_type
    sharding_type = setting.sharding_type
    kernel_type = setting.kernel_type

    backend = setting.backend
    world_size = setting.world_size

    test_case_name = str(sharder_type) + "#" + str(sharding_type) + "#" + backend + "#" + str(world_size) + "#" + str(quantization)
    return test_case_name

def construct_unit_test_str_quant(setting: DistributedConfig, quant_setting: QuantizationConfig):
    quantization = setting.quantization
    sharder_type = setting.sharder_type
    sharding_type = setting.sharding_type
    kernel_type = setting.kernel_type

    backend = setting.backend
    world_size = setting.world_size

    quant_dtype = quant_setting.quant_dtype
    quant_output_dtype = quant_setting.quant_output_dtype

    test_case_name = str(sharder_type) + "#" + str(sharding_type) + "#" + backend + "#" + str(world_size) + "#" + str(quantization) + "#" + str(quant_dtype) + "#" + str(quant_output_dtype)
    return test_case_name

path_main = "/results/outputs"

unit_tests = []
for setting in distributed_settings:
    unit_tests.append(construct_unit_test_str(setting))

for setting in distributed_settings:
    if setting.quantization:
        for quant_setting in quantization_settings:
            unit_tests.append(construct_unit_test_str_quant(setting, quant_setting))

print(len(unit_tests))
    
max_diff = 0
nb_compare = 0
nb_inco = 0
nb_seed = 0
nb_crash = 0
nb_total = 0

crash_list = []
incon_list = []

for seed in range(NUM_MODELS):
    nb_seed += 1
    for input_index in range(NUM_TRAINING_DATASETS):
        result_dict = {}
        standard_unit_test_not_quant = None
        standard_unit_test_quant = None
        for unit_test in unit_tests:
            new_unit_test = "output_" + str(seed) + "_" + str(input_index)+ "_" + unit_test
            unit_test_dir = os.path.join(path_main, "output_" + str(seed), "output_" + str(seed) + "_" + str(input_index), new_unit_test)
            ranks = ["rank_0.p"]
            if len(ranks) == 1:
                try:
                    result_dict[new_unit_test] = load_pred_result_lzma("/results", seed, input_index, unit_test)
                except KeyboardInterrupt:
                    sys.exit(1)
                except:
                    print(traceback.format_exc())
                    print("Result loading failed for ", seed, input_index, " ", new_unit_test)
            else:
                # TODO: handle multiple ranks
                pass
        print("length of result dict: ", len(result_dict))
        if len(result_dict) >= 2:
            for key, item in result_dict.items():
                nb_compare += 1

                if seed >= int(NUM_MODELS/2):
                    value_1_np = normalize_pred(item[0]).detach().cpu().numpy()
                    value_2_np = normalize_pred(item[1]).detach().cpu().numpy()
                else:
                    value_1_np = item[0].detach().cpu().numpy()
                    value_2_np = item[1].detach().cpu().numpy()
                if not np.allclose(value_1_np, value_2_np, atol=5e-4, rtol=1e-4):
                    nb_inco += 1
                    Linf_diff = np.max(np.abs(value_1_np - value_2_np))
                    
                    if Linf_diff > max_diff:
                        max_diff = Linf_diff
                    incon_list.append(["output_" + str(seed), "output_" + str(seed) + "_" + str(input_index), key, key, Linf_diff])

print("Total inconsistencies:", nb_inco)
print("Total comparisons:", nb_compare)

csv_file_path = os.path.join("/results", "csv")
os.makedirs(csv_file_path, exist_ok=True)
with open(os.path.join(csv_file_path, "inconsistency.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "input", "config 1", "config 2", "Linf"])
    writer.writerows(incon_list)

