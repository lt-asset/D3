import os
import keras
import random
import tensorflow_model_optimization as tfmot
from constants import *

random.seed(0)

# DLRM models
for i in range(NUM_MODELS//4, NUM_MODELS//2):
    model_dir = f"../data/models/model_{i}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    frozen_order = [0, 1]

    random.shuffle(frozen_order)
    print(frozen_order)
    with open(model_dir + "frozen_order.txt", 'w') as f:
        f.write(f"{frozen_order}")

# Sequential models
for i in range(3*NUM_MODELS//4, NUM_MODELS):
    model_dir = f"../data/models/model_{i}/"
    model = keras.models.load_model(model_dir + "model.h5")

    frozen_order = []
    for (j, l) in enumerate(model.layers):
        if "input" in l.name or "flatten" in l.name or  "reshape" in l.name:
            continue
        frozen_order.append(j)

    random.shuffle(frozen_order)
    print(frozen_order)
    with open(model_dir + "frozen_order.txt", 'w') as f:
        f.write(f"{frozen_order}")

