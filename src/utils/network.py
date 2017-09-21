import requests


def download_file(url, filename):
  '''
  Generic function to stream-download a large file
  '''
  r = requests.get(url, stream=True)
  with open(filename, 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
      if chunk:
        f.write(chunk)
  return filename
