def build_alts(phrase):
  '''
  Build subphrases from a phrase.
  '''
  words = phrase.split(" ")
  alts = [phrase]
  for j in range(1, min([4, len(words)])):
    for k in range(0, len(words) + 1 - j):
      alts.append(" ".join(words[k:k + j]))
  return alts


def clean_name():
  pass

