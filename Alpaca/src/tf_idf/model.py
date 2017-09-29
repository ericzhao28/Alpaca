import logging
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
from gensim import corpora

from . import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TFIDF():
  def train(self, corpus):
    self.dictionary = Dictionary(corpus)
    self.dictionary.save(config.MODEL_DIR + config.DICT_SAVE)

    raw = [self.dictionary.doc2bow(x) for x in corpus]
    MmCorpus.serialize(config.MODEL_DIR + config.CORPUS_SAVE, raw)
    self.corpus = corpora.MmCorpus(config.MODEL_DIR + config.CORPUS_SAVE)

    self.tfidf = TfidfModel(self.corpus)
    self.tfidf.save(config.MODEL_DIR + config.TFIDF_SAVE)

    return corpus

  def load(self):
    self.dictionary = corpora.Dictionary.load(config.MODEL_DIR + config.DICT_SAVE)
    self.corpus = corpora.MmCorpus(config.MODEL_DIR + config.CORPUS_SAVE)

    self.tfidf = TfidfModel.load(config.MODEL_DIR + config.TFIDF_SAVE)
    return self.corpus, self.tfidf

  def score(self, word):
    return self.tfidf[word]

