from ..src.network import download_file
from . import config

download_file(config.DOWNLOAD_URL, config.DATA_DIR + "file")
