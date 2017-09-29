import tensorflow as tf


class StandardLayers():
  '''
  Standard TF components.
  '''

  def _prediction_layer(self, X, var_scope, config):
    '''
    Predicts end result.
    Args:
      X - input data of shape (batch, features).
      var_scope - string name of tf variable scope.
      config {
          'n_batches': number of batches,
          'n_input': number of input features,
          'n_classes': number of potential output classes
        }
    '''

    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'], config['n_input']))

    with tf.variable_scope(var_scope):
      W = tf.get_variable("W", shape=(config['n_input'], config['n_classes']))
      b = tf.get_variable("bias", shape=(config['n_classes']))
      prediction = tf.nn.softmax(tf.matmul(X, W) + b, name="prediction")
      assert(prediction.shape == (config['n_batches'], config['n_classes']))

      return prediction

  def _dense_layer(self, X, var_scope, config):
    '''
    Predicts end result.
    Args:
      X - input data of shape (batch, features).
      var_scope - string name of tf variable scope.
      config {
          'n_batches': number of batches,
          'n_input': number of input features,
          'n_hidden': number of hidden units,
          'n_output': number of potential output classes
        }
    '''

    assert(type(var_scope) == str)
    assert(type(config) == dict)
    assert(X.shape == (config['n_batches'], config['n_input']))

    with tf.variable_scope(var_scope):
      W_1 = tf.get_variable("W_1", shape=(config['n_input'], config['n_hidden']))
      b_1 = tf.get_variable("bias_1", shape=(config['n_hidden']))
      A = tf.tanh(tf.matmul(X, W_1) + b_1, name="A")

      W_2 = tf.get_variable("W_2", shape=(config['n_hidden'], config['n_output']))
      b_2 = tf.get_variable("bias_2", shape=(config['n_output']))
      output = tf.tanh(tf.matmul(A, W_2) + b_2, name="output")

      assert(output.shape == (config['n_batches'], config['n_output']))
      return output

  def _define_optimization_vars(self, target, prediction, result_weights=None):
    '''
    Defines loss, optim, and various metrics to tarck training progress.
    Args:
      - target - correct labels of shape (batch, classes).
      - prediction - predictions of shape (batch, classes).
      - result_weights - array indicating how much to weight loss for each
                         class, ex: [1, 5].
    Return:
      - loss (tf.float32): regularized loss for pred/target.
      - acc (tf.float32): decimal accuracy.
    '''

    with tf.variable_scope('optimization'):
      regularization = tf.add_n([
          tf.nn.l2_loss(v) for v in tf.trainable_variables()
          if 'bias' not in v.name
      ]) * tf.constant(0.02, dtype=tf.float32)

      delta = tf.constant(0.000001, dtype=tf.float32)
      if result_weights is None:
        loss = regularization - tf.reduce_sum(
            target * tf.log(prediction + delta), name="loss"
        )
      else:
        loss = regularization - tf.reduce_sum(
            target * tf.log(prediction + delta) *
            tf.constant(result_weights, dtype=tf.float32),
            name="loss"
        )

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
      acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

      return loss, acc

  def _summaries(self):
    '''
    Define summaries for tensorboard use.
    '''

    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar("accuracy", self.acc)
      tf.summary.histogram("histogram loss", self.loss)
      summary_op = tf.summary.merge_all()

      return summary_op

