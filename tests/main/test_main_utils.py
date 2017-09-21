from ...src.main import utils


def test_build_alts():
  assert(["woo hugs delic", "woo hugs", "hugs delic", "woo", "hugs", "delic"] == utils.build_alts("woo hugs delic"))


def test_build_candidate_entities():
  def __get_entity(entity):
    return entity in ["i", "food", "like food"]
  assert(["i", "food", "like food"] == utils.build_candidate_entities(__get_entity, "hello i like food"))


def test_score_entities():
  updates = {}

  def __update_entity(entity, key, value, create_if_none=False):
    updates[entity] = (key, value)

  def __tf_idf_scorer(entity, doc):
    denominator = 100.0 if doc == "hello i like food" else 2.0
    if entity in ["hello", "i", "like", "food", "like food"]:
      numerator = 50.0
    else:
      numerator = 5.0
    return numerator / denominator

  utils.build_candidate_entities(__update_entity, __tf_idf_scorer, ["hello i like food"])

  assert(updates['hello'] == ("tfidf", 0.5))
  assert(updates['i'] == ("tfidf", 0.5))
  assert(updates['like'] == ("tfidf", 0.5))
  assert(updates['food'] == ("tfidf", 0.5))

