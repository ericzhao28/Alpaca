from gensim.corpora import TextCorpus
from ...utils import cleaning


class Corpus(TextCorpus):
  def __init__(self, text_stream):
    self.stream = text_stream

  def get_texts(self):
    for text in self.stream:
      try:
        yield [cleaning.clean_name(i) for i in text.split(" ")]
      except AssertionError:
        print("Error")
        continue

  def get_word(self):
    for text in self.stream:
      try:
        yield from [cleaning.clean_name(i) for i in text.split(" ")]
      except AssertionError:
        continue

