from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora import Dictionary

from .config import MODEL_DIR, TFIDF_MODEL, DICT_MODEL

from .model import RatebeerCorpus


def get_model(model_name=TFIDF_MODEL, dictionary_name=DICT_MODEL):
  corpus = RatebeerCorpus()
  corpus.dictionary = Dictionary.load(MODEL_DIR + dictionary_name)

  tfidf = TfidfModel.load(MODEL_DIR + model_name)

  return corpus, tfidf

