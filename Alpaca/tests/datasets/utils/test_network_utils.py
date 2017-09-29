from ....datasets.utils import network_utils
from . import config
import unittest


class TestNetworkUtils(unittest.TestCase):
  def test_download_file(self):
    network_utils.download_file("http://x.com/", config.DUMPS_DIR + "x.html")
    with open(config.DUMPS_DIR + "x.html", 'rb') as myfile:
      data = str(myfile.read().decode())
    self.assertEqual(data, "x")
