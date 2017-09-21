from ...src.utils import network
from . import config


def test_download_file():
  network.download_file("http://x.com/", config.DATA_DIR + config.NETWORK_DATA_NAME)
  with open(config.DATA_DIR + config.NETWORK_DATA_NAME, 'rb') as myfile:
    data = str(myfile.read().decode())
  assert(data == "x")


def test_unzip_file():
  pass
