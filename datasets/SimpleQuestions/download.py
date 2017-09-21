from ..src.utils.network import download_file
from . import config

if __name__ == "__main__":
  download_file(config.DOWNLOAD_URL, config.DATA_DIR + "file")
