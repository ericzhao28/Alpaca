from . import config
from ..utils import csv_utils, shaping_utils
import numpy as np
import csv
from .logger import set_logger


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
  '''

  set_logger.info("Starting preprocessing...")
  return label_data(*segment_data(*load_data(file_path)))


def load_data(file_path):
  '''
  Load in a CSV and parse for data.
  Args:
    - file_path (str): path to raw dataset file.
  Returns:
    - X: np.array([[0.3, 0.3...]...], dtype=float32)
    - metadata: [{info:a, label:x}... (n_classes)]
  '''

  def __featurize_row(row, headers_key):
    '''
    Featurize a row into a real-valued vector.
    '''
    feature_vector = []
    for i, value in enumerate(row):
      if headers_key[i] in config.numerical_fields:
        feature_vector.append(float(value))
    assert(len(feature_vector) == len(config.numerical_fields))
    return np.array(feature_vector, dtype=np.float32)

  def __metadatize_row(row, headers_key):
    '''
    Return metadatized dict for given row.
    '''
    metadatum = {}
    for i, value in enumerate(row):
      if headers_key[i] == config.label_field:
        metadatum['y'] = str(value)
      elif headers_key[i] == config.seq_field:
        metadatum['seq_id'] = str(value)
    return metadatum

  X = []
  metadata = []
  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset csv...")
    for i, row in enumerate(csv.reader(f)):
      if i == 0:
        headers_key = csv_utils.build_headers(row)
        set_logger.debug("Headers key generated: " + str(headers_key))
        continue
      set_logger.debug('Loading row: ' + str(i))
      X.append(__featurize_row(row, headers_key))
      metadata.append(__metadatize_row(row, headers_key))
  set_logger.debug("Basic data loading complete.")
  return np.array(X, dtype=np.float32), metadata


def segment_data(X, metadata):
  '''
  Segment X into sequences based off of metadata.
  Args:
    - X (np.array): data to be segmented on first dim.
    - metadata (list of dicts): metadata used for segmentation,
      namely 'seq_id' in this implementation.
  Returns:
    - new_X (np.array): segmented np.array with +1 rank.
    - new_metadata: metadata cut to align with new_X.
  '''

  set_logger.debug("Segmenting data of shape " + str(X.shape) + "...")
  new_X = []
  new_metadata = []
  seq_map = {}

  set_logger.debug("Segment mapping...")
  for i in range(len(X)):
    seq_id = metadata[i]['seq_id']
    if seq_id not in seq_map:
      seq_map[seq_id] = len(new_X)
      new_X.append([X[i]])
      new_metadata.append(metadata[i])
    else:
      new_X[seq_map[seq_id]].append(X[i])

  set_logger.debug("Average seq len: " + str(len(X) / len(new_X)))

  set_logger.debug("Fixing sequence length padding/cutting...")
  for i in range(len(new_X)):
    new_X[i] = shaping_utils.fix_vector_length(new_X[i], config.SEQ_LEN)

  set_logger.debug("Data segmentation complete.")
  return np.array(new_X, dtype=np.float32), new_metadata


def label_data(X, metadata):
  '''
  Label X based off of metadata.
  Args:
    - X (np.array): data to be labelled.
    - metadata (list of dicts): metadata holding in this case, 'y'.
  Returns:
    - X (np.array): the original X from args.
    - Y (np.array): the new labels for dataset.
  '''

  set_logger.debug("Labelling data...")
  Y = []
  for datum in metadata:
    Y.append(shaping_utils.one_hot(datum['y'], config.label_classes))
  set_logger.debug("Data labelled!")
  return X, np.array(Y, dtype=np.int32)

