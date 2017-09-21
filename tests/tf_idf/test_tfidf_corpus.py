from ...src.tf_idf import Corpus


def test_get_texts():
  corp = Corpus(["hello woo", "What hugs", "yo", "hugs "])
  for i, arr in enumerate(corp.get_texts()):
    if i == 0:
      assert(arr == ["hello", "woo"])
    elif i == 1:
      assert(arr == ["what", "hugs"])
    elif i == 2:
      assert(arr == ["yo"])
    elif i == 3:
      assert(arr == ["hugs"])


def test_get_word():
  corp = Corpus(["hello woo", "What hugs", "yo", "hugs "])
  x = []
  for word in corp.get_word():
    x.append(word)
  assert(x == ["hello", "woo", "what", "hugs", "yo", "hugs"])

