def score_entities(graph, tf_idf_model):
  for entity in graph.entity_names:
    graph.update_entity(entity, "tf_idf", tf_idf_model.score(entity, entity))
    for i in range(0, len(entity), 1):
      subword = entity[i:i+1]
      graph.update_entity(subword, "tf_idf", tf_idf_model.score(subword, entity), create_if_none=True)
    for i in range(0, len(entity), 2):
      subword = entity[i:i+2]
      graph.update_entity(subword, "tf_idf", tf_idf_model.score(subword, entity), create_if_none=True)
    for i in range(0, len(entity), 3):
      subword = entity[i:i+3]
      graph.update_entity(subword, "tf_idf", tf_idf_model.score(subword, entity), create_if_none=True)

def match_entities(graph, entity_text):
  candidates = []
  if graph.get_entity(entity_text):
    candidates.append(entity_text)
  for i in range(0, len(entity), 1):
    subword = entity[i:i+1]
    if graph.get_entity(subword):
      candidates.append(subword)
  for i in range(0, len(entity), 2):
    subword = entity[i:i+2]
    if graph.get_entity(subword):
      candidates.append(subword)
  for i in range(0, len(entity), 3):
    subword = entity[i:i+3]
    if graph.get_entity(subword):
      candidates.append(subword)
  return candidates

