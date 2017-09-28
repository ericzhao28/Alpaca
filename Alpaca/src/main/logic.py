from .utils import build_alts


def build_candidate_entities(graph, entity_text):
  '''
  Identify subword candidates that exist as entities.
  '''
  candidates = []
  for alt in build_alts(entity_text):
    if graph.get_entity(graph.build_node(alt)):
      candidates.append(alt)
  return candidates


def score_entities(graph, tf_idf_scorer, entity_texts):
  '''
  Add TF_IDF scores to entities.
  '''
  for entity_text in entity_texts:
    for alt in build_alts(entity_text):
      graph.update_entity_property(
          graph.build_node(alt), "tfidf", tf_idf_scorer(alt, entity_text))

