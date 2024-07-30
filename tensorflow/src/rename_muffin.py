import os
from constants import *

if __name__ == "__main__":
    cur_dir = os.getcwd()
    model_dir = f"{cur_dir}/../data/models"

    for m in range(NUM_MODELS//2, NUM_MODELS):
        dest_dir = f"{model_dir}/model_{m}"
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

    # Copy muffin models
    muffin_dir = f"{cur_dir}/muffin/data/cifar10_output/"
    for i in range(1, NUM_MODELS//2 + 1):
        src_dir = muffin_dir + str(i).zfill(6) + "/models"

        cur_m = i + NUM_MODELS//2 - 1
        dest_dir = f"{model_dir}/model_{cur_m}"
        if os.path.exists(f"{src_dir}/tensorflow.h5"):
            print(f"Copying {src_dir} to {dest_dir}")
            os.system(f"cp {src_dir}/model.json {dest_dir}/model.json")
            os.system(f"cp {src_dir}/tensorflow.h5 {dest_dir}/model.h5")

        cur_m = i + NUM_MODELS//2 + NUM_MODELS//4 - 1
        dest_dir = f"{model_dir}/model_{cur_m}"
        if os.path.exists(f"{src_dir}/tensorflow.h5"):
            print(f"Copying {src_dir} to {dest_dir}")
            os.system(f"cp {src_dir}/model.json {dest_dir}/model.json")
            os.system(f"cp {src_dir}/tensorflow.h5 {dest_dir}/model.h5")
