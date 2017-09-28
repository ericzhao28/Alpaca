"""
This module offers miscellanious networking functions.
"""

import requests


def download_file(url, file_path):
  '''
  Generic function to stream-download a large file.
  Args:
    - url (str): url to download from.
    - file_path (str): path to save file to.
  Returns:
    - file_path (str): original file_path where saved is.
  '''
  r = requests.get(url, stream=True)
  with open(file_path, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
  return file_path
