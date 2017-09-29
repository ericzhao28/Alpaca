'''
Module for shaping/manipulating data.
'''

import numpy as np


def one_hot(value, candidates):
  '''
  Returns proper one-hot representation of a value.
  Invalid value will raise Value Error.
  Args:
    - value (str): value belonging to candidates.
    - candidates (list of str): possible values.
  Returns:
    - vector (np.arr): one-hot vector represenation.
  '''
  for i, candidate in enumerate(candidates):
    if candidate == value:
      vector = np.zeros(len(candidates))
      vector[i] = 1
      return vector
  raise ValueError("Value not in candidates for one-hot.")


def fix_vector_length(vector, length):
  '''
  Fix a vector to a given length, padding and cutting
  as needed.
  Args:
    - vector (np.arr): input vector.
    - length (int): desired length.
  Returns:
    - vector (np.arr): vector with proper length in dim 1.
  '''
  vector = np.array(vector)
  right_padding = length - min(length, vector.shape[0])
  padding_shape = [[0, right_padding]] + [[0, 0] for _ in range(len(vector.shape) - 1)]
  vector = np.pad(vector[:length], padding_shape,
                  'constant', constant_values=0)
  assert(vector.shape[0] == length)
  return vector

