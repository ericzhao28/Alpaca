def gather_candidate_answers(parent_entity, desired_relation, graph):
  candidate_answers = []
  for child in graph.get_childs(parent_entity, desired_relation):
    candidate_answers.append((child, desired_relation))
  candidate_answers.sort(lambda x: x[0]['tf_idf'])
  return candidate_answers

