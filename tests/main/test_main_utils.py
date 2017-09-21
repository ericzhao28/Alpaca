from ...src.main import utils


def test_build_alts():
  assert(["woo hugs delic", "woo hugs", "hugs delic", "woo", "hugs", "delic"] == utils.build_alts("woo hugs delic"))


def test_build_candidate_entities():
  def __get_entity():
    pass
  assert(True is utils.build_candidate_entities(__get_entity, "hello i like food"))


def test_score_entities():
  def __update_entity():
    pass

  def __tf_idf_scorer():
    pass

  assert(True is utils.build_candidate_entities(__update_entity, __tf_idf_scorer, "hello i like food"))

