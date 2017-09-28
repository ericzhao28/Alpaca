import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


class SequenceLayers():
  '''
  Class containing useful layers for sequential analysis.
  '''

  def _encoder_layer(self, X, var_scope, config):
    '''
    Builds a layer for a simple GRU-encoding of a sequence.
    Args:
      X - input data of shape (batch, seq, unit_features)
      var_scope - string name of tf variable scope
      config {
          'n_batches': number of sequences (batches),
          'n_steps': length of sequence,
          'n_features': number of features,
          'h_gru': hidden units in GRU,
          'h_dense': hidden units in dense
        }
    '''
    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'],
                       config['n_steps'],
                       config['n_features']))

    with tf.variable_scope(var_scope):
      fwd_gru = tf.nn.rnn_cell.GRUCell(config['h_gru'])
      bwd_gru = tf.nn.rnn_cell.GRUCell(config['h_gru'])

      X_unstacked = tf.unstack(tf.transpose(X, (1, 0, 2)), name="X_unstacked")
      H_inv, O_fwd, O_bwd = tf.nn.static_bidirectional_rnn(fwd_gru,
                                                           bwd_gru,
                                                           X_unstacked,
                                                           dtype=tf.float32)
      H = tf.transpose(H_inv, (1, 0, 2), name="H")
      O = tf.concat((O_fwd, O_bwd), axis=1, name="O")

      W = tf.get_variable("W", dtype=tf.float32,
                          shape=(2 * config['h_gru'], config['h_dense']))

      b = tf.get_variable("bias", shape=(config['h_dense']))
      A = tf.tanh(tf.matmul(O, W) + b, name="A")
      assert(A.shape == (config['n_batches'], config['h_dense']))

      return A, H

  def _attention_encoder_layer(self, X, var_scope, config):
    '''
    Builds a layer for a GRU-encoding of a sequence with self attention.
    Args:
      X - input data of shape (batch, seq, unit_features)
      var_scope - string name of tf variable scope
      config {
          'n_batches': number of sequences (batches),
          'n_steps': length of sequence,
          'n_features': number of features,
          'h_gru': hidden units in GRU,
          'h_dense': hidden units in dense
          'h_att': hidden units in attention calc
        }
    '''
    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'],
                       config['n_steps'],
                       config['n_features']))

    with tf.variable_scope(var_scope):
      fwd_gru = tf.nn.rnn_cell.GRUCell(config['h_gru'])
      bwd_gru = tf.nn.rnn_cell.GRUCell(config['h_gru'])

      X_unstacked = tf.unstack(tf.transpose(X, (1, 0, 2)), name="X_unstacked")
      H_inv, _, _ = tf.nn.static_bidirectional_rnn(fwd_gru,
                                                   bwd_gru,
                                                   X_unstacked,
                                                   dtype=tf.float32)
      H = tf.transpose(H_inv, (1, 0, 2), name="H")

      W_s_1 = tf.get_variable("W_s_1", dtype=tf.float32,
                              shape=(2 * config['h_gru'], config['h_att']))
      r_mid = tf.tanh(
          tf.matmul(
              tf.reshape(
                  H,
                  (
                      config['n_batches'] * config['n_steps'],
                      2 * config['h_gru']
                  )
              ),
              W_s_1
          ),
          name="r_mid"
      )
      assert(r_mid.shape ==
             (config['n_batches'] * config['n_steps'], config['h_att']))

      W_s_2 = tf.get_variable("W_s_2", dtype=tf.float32,
                              shape=(config['h_att'], 1))
      r = tf.nn.softmax(
          tf.reshape(
              tf.squeeze(tf.matmul(r_mid, W_s_2)),
              (config['n_batches'], config['n_steps'])
          ),
          name="r"
      )
      assert(r.shape == (config['n_batches'], config['n_steps']))

      M = tf.squeeze(
          tf.matmul(
              tf.transpose(H, (0, 2, 1)),
              tf.expand_dims(r, 2)),
          name="M"
      )
      assert(M.shape == (config['n_batches'], 2 * config['h_gru']))

      W = tf.get_variable("W", dtype=tf.float32,
                          shape=(2 * config['h_gru'], config['h_dense']))
      b = tf.get_variable("bias", shape=(config['h_dense']))
      A = tf.tanh(tf.matmul(M, W) + b, name="A")
      assert(A.shape == (config['n_batches'], config['h_dense']))

      return A, H

  def _embedding_layer(self, X, var_scope, config):
    '''
    Builds a layer for fetching embeddings based on id.
    Args:
      X - input data of shape (batch)
      var_scope - string name of tf variable scope
      config {
          'emb_path': path to embedding word2vec file,
          'emb_dim': dimensions of each word embedding,
        }
    '''

    self.logger.debug('Creating embedding layer...')

    self.logger.debug('Loading word2vec file...')
    model = KeyedVectors.load_word2vec_format(config['emb_path'])

    self.logger.debug('Creating inverse vocab...')
    vocab = {}
    for k, v in model.vocab.items():
      vocab[k] = v.index

    self.logger.debug('Creating embedding tensor...')
    embeddings = np.zeros((len(vocab), config['emb_dim']))
    for k, v in model.vocab.items():
      embeddings[v.index] = model[k]
    del(model)
    del(vocab)

    self.logger.debug('Defining TF embedding look up layer...')
    embedded_X = tf.cast(
        tf.nn.embedding_lookup(embeddings, X),
        name="embedded_x",
        dtype=tf.float32
    )

    self.logger.debug('Embedding layer created.')
    return embedded_X

