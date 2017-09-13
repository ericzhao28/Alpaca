def gather_candidate_answers(first_entity, desired_relation, graph):
  candidate_answers = []
  for neighbor, relation in graph.reachable_entities(entity):
    if desired_relation == relation:
      candidate_answers.append((neighbor, relation))
  return candidate_answers

def select_answer(second_entity, candidate_answers):
  def scoring_func(candidate, second_entity):
    return cosine_similarity(candidate['text'], second_entity['text'])
  return sort(candidate_answers, scoring_func, second_entity)

