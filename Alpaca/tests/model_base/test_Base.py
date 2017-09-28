from ...src.model_base import Base
from .logger import test_logger
import numpy as np
import tensorflow as tf


def test_shuffle_and_partition():
  with tf.Session() as sess:
    model = Base(sess, None, test_logger)
    X = np.reshape(np.arange(10000), (100, 10, 10))
    Y = np.reshape(np.arange(200), (100, 2))
    dataset = model.shuffle_and_partition(X, Y, 10, 20)
    assert(dataset['train']['X'].shape == (70, 10, 10))
    assert(dataset['train']['Y'].shape == (70, 2))
    assert(dataset['test']['X'].shape == (10, 10, 10))
    assert(dataset['test']['Y'].shape == (10, 2))
    assert(dataset['val']['X'].shape == (20, 10, 10))
    assert(dataset['val']['Y'].shape == (20, 2))
    flattened_X = np.concatenate([
        dataset['train']['X'],
        dataset['test']['X'],
        dataset['val']['X'], 
    ]).flatten()
    assert(len(set(flattened_X)) == len(flattened_X))
    flattened_Y = np.concatenate([
        dataset['train']['Y'],
        dataset['test']['Y'],
        dataset['val']['Y'],
    ]).flatten()
    assert(len(set(flattened_Y)) == len(flattened_Y))

