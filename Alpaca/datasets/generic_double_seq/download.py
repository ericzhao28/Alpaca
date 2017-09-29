from . import config, preprocess_file
import pickle
from .logger import set_logger


def main():
  '''
  Main function of this download module.
  Attempts to load existing dataset and if fails, downloads raw dataset and
  saves a preprocessed copy.
  '''

  try:
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, "rb") as f:
      set_logger.info("Dataset already exists. Attempting pickle load...")
      pickle.load(f)
      set_logger.info("Dataset pickle loaded")

  except (EOFError, OSError, IOError) as e:
    set_logger.info("No dataset yet. Creating and writing new dataset...")
    with open(config.DUMPS_DIR + config.PROCESSED_SAVE_NAME, 'wb') as f:
      pickle.dump(preprocess_file(config.DUMPS_DIR + config.RAW_SAVE_NAME), f)
      set_logger.info("Dataset pickle loaded and dumped.")

  return None


if __name__ == "__main__":
  set_logger.info("Beginning dataset download")
  main()
