from ..datasets import SimpleQuestions
from ..models.tf_idf import TF_IDF
from ..models.tensorflow_query_structuring import Parser

from .entity_linking import match_entities, score_entities
from .answer_selection import gather_candidate_answers
from ..utils import cleaning


def initialize():
  # Initialize Knowledge Graph
  graph = SimpleQuestions.load_graph()

  tf_idf_model = TF_IDF.Model()
  tf_idf_model.load()
  score_entities(graph, tf_idf_model)

  local_parser = Parser()
  local_parser.build_model()

  def execute(query):
    query = [cleaning.clean_name(x) for x in query.split(" ")]
    entity_inds = local_parser.predict_entities(query)
    full_entity = " ".join([query[i] for i in entity_inds])
    predicted_relationship = local_parser.predict_relationship(query)

    base_entity_candidates = match_entities(graph, full_entity)

    candidate_answers = []
    for base_entity in base_entity_candidates:
      candidate_answers += gather_candidate_answers(base_entity, predicted_relationship, graph)
    return candidate_answers[0]

  return execute


def train():
  graph = SimpleQuestions.create_graph()

  tf_idf_corpus = TF_IDF.Corpus(graph.get_all_relations)
  tf_idf_model = TF_IDF.Model()
  tf_idf_model.train(tf_idf_corpus)

  local_parser = Parser()
  local_parser.train()
