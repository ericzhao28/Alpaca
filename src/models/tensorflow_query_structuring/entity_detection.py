given questions
k = 2

embeddings = embed(question)
hiddens = rnn(embeddings, rnn_hidden)
W = (rnn_hidden, k)
boops = tf.reshape(final_state, W)
boop_probs = unstack(softmax(boop, axis=-1))

for prob in boop_probs:
  is_context = bool(argmax(prob))
