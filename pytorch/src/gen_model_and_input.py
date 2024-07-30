import argparse
import os

from model_generation import generate_models
from input_generation import generate_inputs, generate_sequential_input
from dlrm_model_generation import TestModel
from constants import BATCH_SIZE, NUM_MODELS, NUM_TRAINING_DATASETS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action='store_true',
                        help='whether to regenerate models and inputs even if they already exist')
    args = parser.parse_args()
    config = vars(args)

    num_models = NUM_MODELS
    num_inputs = NUM_TRAINING_DATASETS
    regen = config["regen"]

    save_base_dir = "/data"
    model_save_base_dir = os.path.join(save_base_dir, "models")
    os.makedirs(model_save_base_dir, exist_ok=True)
    for i in range(num_models):
        seed_model = i
        model_save_path = os.path.join(model_save_base_dir, "model_{}".format(i))
        if i < int(num_models/4):
            model = generate_models(model_class=TestModel, seed=seed_model, model_save_path=model_save_path, regen=regen)
        if i >= int(num_models/4) and i < int(num_models/2):
            src_model = os.path.join(model_save_base_dir, "model_{}".format(i-int(num_models/4)))
            dst_model = model_save_path
            os.system("cp {} {}".format(src_model, dst_model))

        input_save_base_dir = os.path.join(save_base_dir, "inputs", "input_{}".format(i))
        os.makedirs(input_save_base_dir, exist_ok=True)
        for j in range(num_inputs):
            seed_input = i*num_inputs + j
            input_save_path = os.path.join(input_save_base_dir, "input_{}_{}".format(i, j))
            if i < int(num_models/4):
                inputs = generate_inputs(model=model, input_seed=seed_input, batch_size=BATCH_SIZE, input_save_path=input_save_path, regen=regen)
            elif i >= int(num_models/4) and i < int(num_models/2):
                src_input = os.path.join(save_base_dir, "inputs", "input_{}".format(i-int(num_models/4)), "input_{}_{}".format(i-int(num_models/4), j))
                dst_input = input_save_path
                os.system("cp {} {}".format(src_input, dst_input))
                print("cp {} {}".format(src_input, dst_input))
            elif i >= int(num_models/2) and i < int(3*num_models/4):
                inputs = generate_sequential_input(input_seed=seed_input, batch_size=BATCH_SIZE, input_save_path=input_save_path, regen=regen)
            else:
                src_input = os.path.join(save_base_dir, "inputs", "input_{}".format(i-int(num_models/4)), "input_{}_{}".format(i-int(num_models/4), j))
                dst_input = input_save_path
                print("cp {} {}".format(src_input, dst_input))
                os.system("cp {} {}".format(src_input, dst_input))


