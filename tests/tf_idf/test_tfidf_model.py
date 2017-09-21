from ...src.tf_idf import Model, Corpus


def test_tfidf():
  corp = Corpus(["hello woo", "What hugs", "yo", "hugs "])
  tfidf = Model()
  tfidf.train(corp)

  tfidf2 = Model()
  tfidf2.load()

  assert(tfidf.score("yo", "woo yo") == 0.4)
  assert(tfidf2.score("woo", "woo yo") == 0.3)

