from ....datasets.utils import csv_utils
import unittest


class TestCSVUtils(unittest.TestCase):
  def test_build_headers(self):
    row = ["chocolate", "yummy", "weird"]
    headers_key = csv_utils.build_headers(row)
    self.assertEqual(headers_key, {0: "chocolate", 1: "yummy", 2: "weird"})

