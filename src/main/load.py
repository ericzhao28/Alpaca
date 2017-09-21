from ..datasets import SimpleQuestions
from ..models import Parser
from .. import utils
from .. import graph_layer
from .. import tf_idf

from .utils import build_candidate_entities, score_entities


def initialize():
  '''
  Initialize answer_query function.
  '''

  # Load tf_idf
  tf_idf_model = tf_idf.model.Model()
  tf_idf_model.load()

  # Load graph
  graph = graph_layer.load_graph()
  for entity in graph.entity_names:
    score_entities(graph, tf_idf_model.score, entity)

  # Load local_parser
  local_parser = Parser()
  local_parser.initialize()
  local_parser.restore()

  def answer_query(query):
    '''
    Generate function to answer queries with the best answer.
    '''
    query = [utils.cleaning.clean_name(x) for x in query.split(" ")]

    # Candidate entities
    entity_text = ""
    for i, score in enumerate(local_parser.predict_entities(query)):
      if score == 1:
        entity_text += " " + query[i]
    entity_candidates = build_candidate_entities(graph.get_entity, entity_text)

    # Relationship prediction
    predicted_relationship = local_parser.predict_relationship(query)

    # Candidate answers
    answers = []
    for candidate in entity_candidates:
      for child in graph.get_childs(candidate, predicted_relationship):
        answers.append((child, predicted_relationship))
    return answers.sort(lambda x: x[0]['tf_idf'])[0]

  return answer_query


def train():
  '''
  Get things started by training graph,
  tf_idf and local_parser
  '''

  # Create graph
  graph = graph_layer.create_graph(SimpleQuestions.load())

  # Train tf_idf
  tf_idf_corpus = tf_idf.Corpus(graph.get_all_relations())
  tf_idf_model = tf_idf.Model()
  tf_idf_model.train(tf_idf_corpus)

  # Train local parser
  local_parser = Parser()
  local_parser.initialize()
  local_parser.train()
  local_parser.save()

