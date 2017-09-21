import logging
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora import Dictionary

from . import config
from . import Corpus as DefaultCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Model():
  def train(self, corpus):
    corpus.dictionary = Dictionary(corpus.get_texts())
    self.tfidf = TfidfModel(corpus)
    corpus.dictionary.save(config.MODEL_DIR + config.DICT_SAVE)
    self.tfidf.save(config.MODEL_DIR + config.TFIDF_SAVE)
    return corpus

  def load(self):
    corpus = DefaultCorpus.Corpus()
    corpus.dictionary = Dictionary.load(config.MODEL_DIR + config.DICT_SAVE)
    self.tfidf = TfidfModel.load(config.MODEL_DIR + config.TFIDF_SAVE)
    return corpus, self.tfidf

  def score(self, word):
    return self.tfidf[word]

