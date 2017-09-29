from ..models import EntModel, RelModel
from ..tf_idf import TFIDF
from ..graph.load import load_graph
from . import rel_config, ent_config
from .logic import score_entities, build_candidate_entities
from .utils import clean_name


def initialize_func(tf_sess, neo4j_sess, logger):
  '''
  Initialize main function to predict using model.
  '''

  # Initialize rel model
  rel_model = RelModel(tf_sess, rel_config, logger)
  logger.debug('Rel Model instantiated.')
  rel_model.initialize()
  logger.debug('Rel Model initialized.')
  rel_model.restore()
  logger.debug('Rel Model restored.')

  # Initialize ent model
  ent_model = EntModel(tf_sess, ent_config, logger)
  logger.debug('Ent Model instantiated.')
  ent_model.initialize()
  logger.debug('Ent Model initialized.')
  ent_model.restore()
  logger.debug('Ent Model restored.')

  # Initialize tf_idfs
  tf_idf_model = TFIDF()
  logger.debug('TF_IDF Model instantiated.')
  tf_idf_model.load()
  logger.debug('TF_IDF Model loaded.')

  # Initialize neo4j
  graph = load_graph(neo4j_sess)
  logger.debug('Neo4j graph loaded.')
  for entity in graph.entity_names:
    score_entities(graph, tf_idf_model.score, entity)
  logger.debug('Entities scored and graph fully loaded.')

  def function(query):
    '''
    Generate function to answer queries with the best answer.
    '''

    # Handle query
    query = [clean_name(x) for x in query.split(" ")]
    logger.debug('Cleaned query: ' + str(query))

    # Predicted entities
    predicted_entities = ent_model.predict([query])[0]
    logger.debug('Ent Model predicted: ' + str(predicted_entities))

    # Relationship prediction
    predicted_relationship = rel_model.predict([query])[0]
    logger.debug('Rel Model predicted: ' + str(predicted_relationship))

    # Find which words in query are entities
    entity_text = ""
    for i, score in enumerate(predicted_entities):
      if score == 1:
        entity_text += " " + query[i]
    entity_text = entity_text.strip()
    entity_candidates = build_candidate_entities(graph, entity_text)

    # Candidate answers
    answers = []
    for candidate in entity_candidates:
      for child in graph.get_childs(candidate, predicted_relationship):
        answers.append((child, predicted_relationship))
    return answers.sort(lambda x: x[0]['tf_idf'])[0]

  return function

