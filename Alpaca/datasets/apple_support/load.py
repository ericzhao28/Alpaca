from . import config
from .logger import set_logger
import pickle


def load(spec):
  '''
  Loads preprocessed data dump if possible.
  '''
  try:
    if spec == "ent":
      with open(config.DUMPS_DIR + config.PROCESSED_ENT_SAVE_NAME, "rb") as f:
        set_logger.info("Dataset ent exists. Attempting pickle load...")
        return pickle.load(f)
    if spec == "rel":
      with open(config.DUMPS_DIR + config.PROCESSED_REL_SAVE_NAME, "rb") as f:
        set_logger.info("Dataset rel exists. Attempting pickle load...")
        return pickle.load(f)

  except (EOFError, OSError, IOError) as e:
    set_logger.info("Dataset does not exist. Returning None.")
    return None

