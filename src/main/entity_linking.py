def score_entities(knowledge_graph):
  new_graph = []
  for entity in knowledge_graph:
    for sub in entity_perm:
      new_graph.append(sub, len(sub), tf_idf(sub, entity.text))
  return new_graph.sort(lambda x: x[1])


def match_entities(entity_text, knowledge_graph):
  candidates = []
  for sub, entity in entity_perm:
    if match:
      candidates.append((sub, entity))
  return candidates


