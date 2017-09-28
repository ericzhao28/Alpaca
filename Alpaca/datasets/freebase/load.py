from . import config
from .logger import set_logger
import pickle


def load():
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    set_logger.info("Dataset exists. Attempting pickle load...")
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, "rb") as f:
      return pickle.load(f)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None

