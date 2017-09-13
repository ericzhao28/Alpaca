class AnswerSelection():
  def __init__(self):
    pass

  def build_model(self):
    input = tf.get_variable("x", [config.batch_size, seq_len])
    word_embeddings = tf.get_variable(“word_embeddings”, [config.vocabulary_size, config.embedding_size])
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
    embedding_layer = []
    final_state = rnn()
    pass

  def train(self):
    pass

  def predict(self):
    pass

  def save(self):
    pass

  def load(self):
    pass

given questions
k = 1837

embeddings = embed(question)
final_state = rnn(embeddings, rnn_hidden)
W = (rnn_hidden, k
boop = (final_state, W)
probs = softmax(boop)

relation_cls = argmax(probs, axis=1)
