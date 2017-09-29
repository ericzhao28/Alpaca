from ....datasets.utils import shaping_utils
import numpy as np


def def_one_hot():
  assert(shaping_utils.one_hot("hello",
                               ["hi", "hello", "hugs"]) == np.array([0, 1, 0]))
  assert(shaping_utils.one_hot("hi",
                               ["hi", "hello", "hugs"]) == np.array([1, 0, 0]))
  failed = False
  try:
    shaping_utils.one_hot("lonely", ["hi", "hello", "hugs"])
  except ValueError:
    failed = True
  assert(failed)


def def_fix_vector_length():
  assert(np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]) ==
         shaping_utils.fix_vector_length(np.full((2, 3), fill_value=1), 4))
  assert(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]) ==
         shaping_utils.fix_vector_length(np.full((7, 3), fill_value=1), 4))
  assert(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0], [0, 0]]]) ==
      shaping_utils.fix_vector_length(np.full((3, 4, 2), fill_value=1), 4))
  assert(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[1, 1], [1, 1], [1, 1], [1, 1]]]) ==
      shaping_utils.fix_vector_length(np.full((5, 4, 2), fill_value=1), 3))
  assert(np.array([
      [[1, 1], [1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1], [1, 1]],
      [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]]) ==
      shaping_utils.fix_vector_length(np.full((2, 4, 2), fill_value=1), 6))
