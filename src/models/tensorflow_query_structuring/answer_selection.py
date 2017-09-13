given questions
k = 1837

embeddings = embed(question)
final_state = rnn(embeddings, rnn_hidden)
W = (rnn_hidden, k
boop = (final_state, W)
probs = softmax(boop)

relation_cls = argmax(probs, axis=1)
