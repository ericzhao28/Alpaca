import tensorflow as tf
from . import Base_Model


class Layered_Model(Base_Model.Base_Model):

  def _encoder_layer(self, X, var_scope, config):
    '''
    Builds a layer for a simple GRU-encoding of a sequence.
    Args:
      X - input data of shape (batch, seq, unit_features)
      var_scope - string name of tf variable scope
      config {
          'n_seqs': number of sequences (batches),
          'seq_len': length of sequence,
          'n_features': number of features,
          'n_gru_hidden': hidden units in GRU,
          'n_dense_hidden': hidden units in dense
        }
    '''
    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_seqs'], config['seq_len'], config['n_features']))

    with tf.variable_scope(var_scope):
      fwd_gru = tf.nn.rnn_cell.GRUCell(config['n_gru_hidden'])
      bwd_gru = tf.nn.rnn_cell.GRUCell(config['n_gru_hidden'])

      X_unstacked = tf.unstack(tf.transpose(X, (1, 0, 2)), name="X_unstacked")
      H_inverse, O_fwd, O_bwd = tf.nn.static_bidirectional_rnn(fwd_gru, bwd_gru, X_unstacked, dtype=tf.float32)
      H = tf.transpose(H_inverse, (1, 0, 2), name="H")

      O = tf.concat((O_fwd, O_bwd), axis=1)
      W = tf.get_variable("W", shape=(2 * config['n_gru_hidden'], config['n_dense_hidden']), dtype=tf.float32)
      A = tf.tanh(tf.matmul(O, W), name="A_unshaped")

      return A, H

  def _attention_encoder_layer(self, X, var_scope, config):
    '''
    Builds a layer for a GRU-encoding of a sequence with self attention.
    Args:
      X - input data of shape (batch, seq, unit_features)
      var_scope - string name of tf variable scope
      config {
          'n_seqs': number of sequences (batches),
          'seq_len': length of sequence,
          'n_features': number of features,
          'n_gru_hidden': hidden units in GRU,
          'n_dense_hidden': hidden units in dense,
          'n_attention_hidden': hidden units in attention calc
        }
    '''
    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_seqs'], config['seq_len'], config['n_features']))

    with tf.variable_scope(var_scope):
      fwd_gru = tf.nn.rnn_cell.GRUCell(config['n_gru_hidden'])
      bwd_gru = tf.nn.rnn_cell.GRUCell(config['n_gru_hidden'])

      X_unstacked = tf.unstack(tf.transpose(X, (1, 0, 2)), name="X_unstacked")
      H_inverse, _, _ = tf.nn.static_bidirectional_rnn(fwd_gru, bwd_gru, X_unstacked, dtype=tf.float32)
      H = tf.transpose(H_inverse, (1, 0, 2), name="H")

      W_s_1 = tf.get_variable("W_s_1", shape=(2 * config['n_gru_hidden'], config['n_attention_hidden']))
      W_s_2 = tf.get_variable("W_s_2", shape=(config['n_attention_hidden'], 1))

      r_mid = tf.tanh(tf.matmul(tf.reshape(H, (config['n_seqs'] * config['seq_len'], 2 * config['n_gru_hidden'])), W_s_1), name="r_mid")
      r = tf.nn.softmax(tf.reshape(tf.squeeze(tf.matmul(r_mid, W_s_2)), (config['n_seqs'], config['seq_len'])), name="r")

      M = tf.squeeze(tf.matmul(tf.transpose(H, (0, 2, 1)), tf.expand_dims(r, 2)), name="M")
      W = tf.get_variable("W", shape=(2 * config['n_gru_hidden'], config['n_dense_hidden']), dtype=tf.float32)
      A = tf.tanh(tf.matmul(M, W), name="A_unshaped")
      assert(A.shape == (config['n_seqs'], config['n_dense_hidden']))

      return A, H

  def _prediction_layer(self, X, var_scope, config):
    '''
    Predicts end result
    Args:
      X - input data of shape (batch, features)
      var_scope - string name of tf variable scope
      config {
          'n_batches': number of batches,
          'n_input': number of input features,
          'n_classes': number of potential output classes,
        }
    '''
    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'], config['n_input']))

    with tf.variable_scope(var_scope):
      W = tf.get_variable("W", shape=(config['n_input'], config['n_classes']))
      prediction = tf.nn.softmax(tf.matmul(X, W), name="prediction")
      assert(prediction.shape == (config['n_batches'], config['n_classes']))

      return prediction

  def _define_optimization_vars(self, target, prediction, result_weights):
    '''
    Defines loss, optim, and various metrics to tarck training progress
    Args:
      target - correct labels of shape (batch, classes)
      prediction - predictions of shape (batch, classes)
      result_weights - array indicating how much to weight loss for each class,
                       ex: [1, 5]
    '''
    with tf.variable_scope('optimization'):
      loss = -tf.reduce_sum(target * tf.log(prediction) * tf.constant(result_weights, dtype=tf.float32), name="loss")

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
      acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

      return loss, acc

  def _summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar("accuracy", self.acc)
      tf.summary.histogram("histogram loss", self.loss)
      summary_op = tf.summary.merge_all()

      return summary_op

