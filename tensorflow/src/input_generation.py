import tensorflow as tf
import numpy as np
import pickle
import os
from typing import *
from constants import *
from tensorflow import Tensor


def generate_DLRM_input(input_path: str, size_dense: int, embed_vocab_size: List[int]):

    # Generate one input tuple (dense_input, sparse_input)
    def gen_one(size_dense: int, embed_dims: List[int]):
        # Dense input
        input_dense = tf.random.uniform(
            shape=[size_dense], minval=0, maxval=1, dtype=tf.float32)

        # Sparse input
        lst_input_sparse: List[Tensor] = []
        for (_, n) in enumerate(embed_dims):
            lst_input_sparse.append(tf.random.uniform(
                shape=[1, 1], minval=0, maxval=n, dtype=tf.int32))

        input_sparse = tf.concat(lst_input_sparse, axis=0)

        return input_dense, input_sparse

    # Training data
    for i in range(NUM_TRAINING_DATASETS):
        train_dense = []
        train_sparse = []
        train_label: Tensor = tf.random.uniform(shape=(BATCH_SIZE, 1))
        for _ in range(BATCH_SIZE):
            d, s = gen_one(size_dense, embed_vocab_size)
            train_dense.append(d)
            train_sparse.append(s)
        # Save training inputs
        train_tuple = (train_dense, train_sparse, train_label)
        with open(f"{input_path}/train_{i}.pk", 'wb') as f:
            pickle.dump(train_tuple, f)

    # Test data
    for i in range(NUM_TRAINING_DATASETS):
        test_dense = []
        test_sparse = []
        test_label: Tensor = tf.random.uniform(shape=(BATCH_SIZE, 1))
        for _ in range(BATCH_SIZE):
            d, s = gen_one(size_dense, embed_vocab_size)
            test_dense.append(d)
            test_sparse.append(s)
        # Save test inputs
        test_tuple = (test_dense, test_sparse, test_label)
        with open(f"{input_path}/test_{i}.pk", 'wb') as f:
            pickle.dump(test_tuple, f)


def generate_Sequential_input(input_path: str):
    # Training data
    for i in range(NUM_TRAINING_DATASETS):
        train_input: Tensor = tf.random.uniform(shape=(BATCH_SIZE, 32, 32, 3))
        train_label: Tensor = tf.one_hot(tf.random.uniform(
            shape=(BATCH_SIZE,), minval=0, maxval=10, dtype=tf.int32), 10)
        # Save training inputs
        train_tuple = (train_input, train_label)
        with open(f"{input_path}/train_{i}.pk", 'wb') as f:
            pickle.dump(train_tuple, f)

    # Test data
    for i in range(NUM_TRAINING_DATASETS):
        test_input: Tensor = tf.random.uniform(shape=(BATCH_SIZE, 32, 32, 3))
        test_label: Tensor = tf.one_hot(tf.random.uniform(
            shape=(BATCH_SIZE,), minval=0, maxval=10, dtype=tf.int32), 10)
        # Save test inputs
        test_tuple = (test_input, test_label)
        with open(f"{input_path}/test_{i}.pk", 'wb') as f:
            pickle.dump(test_tuple, f)
