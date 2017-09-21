def build_alts(phrase):
  '''
  Build subphrases from a phrase.
  '''
  words = phrase.split(" ")
  alts = [phrase]
  for j in range(1, 4):
    for k in range(0, len(words), j):
      alts.append(" ".join(words[k:k + j]))
  return alts


def build_candidate_entities(get_entity, entity_text):
  '''
  Identify subword candidates that exist as entities.
  '''
  candidates = []
  for alt in build_alts(entity_text):
    if get_entity(alt):
      candidates.append(alt)
  return candidates


def score_entities(update_entity, tf_idf_scorer, entity_texts):
  '''
  Add TF_IDF scores to entities.
  '''
  for entity_text in entity_texts:
    for alt in build_alts(entity_text):
      update_entity(alt, "tfidf", tf_idf_scorer(alt, entity_text),
                    create_if_none=True)

