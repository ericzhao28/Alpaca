'''
Module for csv functions.
'''


def build_headers(row):
  '''
  Build headers for a row.
  Args:
    - row (list): from csv.
  Returns:
    - headers_key (dict): {0: "ip", 1: "name"}
  '''
  headers_key = {}
  for j, field in enumerate(row):
    headers_key[j] = field
  return headers_key

