import json
import numpy as np
from .logger import set_logger


def preprocess_file(file_path):
  '''
  Return preprocessed dataset from raw data file.
  Returns:
    - dataset (list): X (np.arr), Y (np.arr)
  '''

  set_logger.info("Starting preprocessing...")
  return load_data(file_path)


def load_data(file_path):
  '''
  Load in a JSON and parse for data.
  Args:
    - file_path (str): path to raw dataset file.
  Returns:
    - X: np.array([[0.3, 0.3...]...], dtype=float32)
    - metadata: [{info:a, label:x}... (n_classes)]
  '''

  def __clean_string(m):
    return m.strip().decode('unicode_escape').encode('ascii', 'ignore')

  def __parse_message(message):
    return " ".join([__clean_string(m) for m in message])

  X = []
  Y_ent = []
  Y_rel = []
  with open(file_path, 'r') as f:
    set_logger.debug("Opened dataset json...")
    for line in f.readlines():
      parsed = json.loads(line)
      X.append(__parse_message(parsed['question_description']))
      convo = []
      for post in parsed['reply_list']:
        convo.append(post['replier_description'])
      Y_rel.append(Y_rel.append(parsed['question_breadcrumb']))
      Y_ent.append([__parse_message(message) for message in convo])  # Placeholder, this is broken

  set_logger.debug("Basic data loading complete.")
  return np.array(X), np.array(Y_ent), np.array(Y_rel)

