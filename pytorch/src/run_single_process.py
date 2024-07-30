import os
import time
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from dlrm_model_generation import TestModel
from worker_quant import main_test_quant
from config import distributed_settings, quantization_settings
from constants import BATCH_SIZE, NUM_MODELS, NUM_TRAINING_DATASETS


if __name__ == "__main__":
    num_models = NUM_MODELS
    num_inputs = NUM_TRAINING_DATASETS

    start = time.time()

    procs = []
    for i in range(num_models):
        seed = i

        model_state_dict_path = os.path.join("/data", "models", "model_{}".format(seed))
        if not os.path.exists(model_state_dict_path):
            model_state_dict_path = os.path.join("/data", "models", "model_{}.onnx".format(seed))

        for j in range(num_inputs):
            input_seed = j
            input_path = os.path.join("/data", "inputs", "input_{}".format(seed), "input_{}_{}".format(seed, input_seed))
            for setting in distributed_settings:
                try:
                    if not setting.quantization:
                        continue
                    quantization = setting.quantization
                    sharder_type = setting.sharder_type
                    sharding_type = setting.sharding_type
                    kernel_type = setting.kernel_type
                    backend = setting.backend
                    world_size = setting.world_size
                    quant_dtype = None
                    quant_output_dtype = None
                    if quantization:
                        sharders = [QuantEmbeddingBagCollectionSharder()]
                        for quant_setting in quantization_settings:
                            quant_dtype = quant_setting.quant_dtype
                            quant_output_dtype = quant_setting.quant_output_dtype

                            arguments = {}
                            arguments["sharders"] = sharders
                            arguments["backend"] = backend
                            arguments["world_size"] = world_size
                            arguments["model_class"] = TestModel
                            arguments["test_case_name"] = str(sharder_type) + "#" + str(sharding_type) + "#" + backend + "#" + str(world_size) + "#" + str(quantization) + "#" + str(quant_dtype) + "#" + str(quant_output_dtype)
                            print(arguments["test_case_name"])
                            arguments["model_state_dict_path"] = model_state_dict_path
                            arguments["input_path"] = input_path
                            arguments["seed"] = seed
                            arguments["input_seed"] = input_seed
                            arguments["global_batch_size"] = BATCH_SIZE
                            arguments["quantization"] = quantization
                            arguments["quant_dtype"] = quant_dtype
                            arguments["quant_output_dtype"] = quant_output_dtype
                            main_test_quant(**arguments)
                    else:
                        pass
                except Exception as e:
                    print(e)
    
    end = time.time()
    print("total time: ", end - start)
