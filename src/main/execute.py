from ..datasets import SimpleQuestions
from ..models.tf_idf import TF_IDF
from ..models.tensorflow_query_structuring import EntityDetector, RelationshipClassifier

from .entity_linking import match_entities, score_entities
from .answer_selection import gather_candidate_answers


def initialize():
  # Initialize Knowledge Graph
  graph = SimpleQuestions.load_graph()

  tf_idf_model = TF_IDF.Model()
  tf_idf_model.load()
  score_entities(graph, tf_idf_model)

  entity_detector = EntityDetector()
  entity_detector.load()

  relationship_clasifier = RelationshipClassifier()
  relationship_clasifier.load()

  def execute(query):
    entity_inds = entity_detector.predict(query)
    full_entity = "".join([query[i] for i in entity_inds])
    predicted_relationship = relationship_clasifier.predict(query)

    base_entity_candidates = match_entities(graph, full_entity)
    # We assume the largest phrase match is the most likely

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

  entity_detector = EntityDetector()
  entity_detector.train()

  relationship_clasifier = RelationshipClassifier()
  relationship_clasifier.train()
