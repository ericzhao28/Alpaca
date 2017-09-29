import tensorflow as tf
import numpy as np
from .standard_layers import StandardLayers
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(4)
np.random.seed(4)


class Base(StandardLayers):
  '''
  Base model featuring useful Tensorflow utilities.
  '''

  def __init__(self, sess, config, logger):
    '''
    Initiate base model.
    Args:
      - sess: tf.Session().
      - config: module with model config.
      - logger: custom logger handler.
    '''

    self.sess = sess
    self.config = config
    self.saver = None
    self.logger = logger
    self.model_name = "default.model"
    self.global_step = tf.Variable(0,
                                   dtype=tf.int32,
                                   trainable=False,
                                   name='global_step')

  def initialize(self):
    '''
    Initializes model.
    Builds model -> starts summary writer -> global vars init.
    '''

    self.logger.debug('Initializing model...')
    self.build_model()

    self.logger.debug('Model built. Initializing model writer...')
    self.train_writer = tf.summary.FileWriter(self.config.GRAPHS_TRAIN_DIR,
                                              self.sess.graph)
    self.test_writer = tf.summary.FileWriter(self.config.GRAPHS_TEST_DIR,
                                             self.sess.graph)

    self.logger.debug('Writer initialized. Initializing TF graph...')
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()
    self.logger.debug('TF graph initialized.')

    self.logger.info('Model initialized.')

  def save(self, global_step=None):
    '''
    Save the current variables in graph.
    Optional option to save for global_step (used in Train).
    '''

    self.logger.debug('Saving model...')
    if self.saver is None:
      self.logger.debug('Saver not initiated, creating new model Saver.')
      self.saver = tf.train.Saver(tf.global_variables())

    if global_step is None:
      self.saver.save(self.sess,
                      self.config.CHECKPOINTS_DIR + self.model_name)
      self.logger.debug('Saved with no global step.')
    else:
      self.saver.save(self.sess,
                      self.config.CHECKPOINTS_DIR + self.model_name,
                      global_step=self.global_step)
      self.logger.debug('Saved with global step.')

    self.logger.info('Model saved.')

  def restore(self, resume=True):
    '''
    Restore TF computation graph from saved checkpoint.
    Args:
      - resume (bool): resume last checkpoint or restore standard
                       save file.
    '''

    self.logger.debug('Restoring model...')

    if self.saver is None:
      self.logger.debug('Saver not initiated, creating new model Saver.')
      self.saver = tf.train.Saver(tf.global_variables())

    if resume:
      self.logger.debug('Resume enabled. Finding newest model checkpoint.')

      ckpt = tf.train.latest_checkpoint(self.config.CHECKPOINTS_DIR)
      if ckpt:
        self.logger.debug('Model checkpoint found. Restoring...')
        self.saver.restore(self.sess, ckpt)
        self.logger.info('Model restored. Resuming from checkpoint.')
        return True
      else:
        self.logger.error('Resume enabled but no model checkpoints found. \
                           \n Terminating...')
        raise ValueError()
    else:
      self.logger.debug('Resume disabled. Restoring from default save...')
      self.saver.restore(
          self.sess, self.config.CHECKPOINTS_DIR + self.model_name)
      self.logger.info('Model restored.')

  def train(self, X, Y, test_X, test_Y):
    '''
    Run model training. Model must have been initialized.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
      Y (np.arr): labels. Assuming len(Y) > batch size.
    '''

    self.logger.info('Starting model training...')

    for j in range(self.config.ITERATIONS):
      for i in range(0, len(X) + 1 - self.config.BATCH_SIZE,
                     self.config.BATCH_SIZE):
        feed_dict = {
            self.x: X[i:i + self.config.BATCH_SIZE],
            self.target: Y[i:i + self.config.BATCH_SIZE]
        }
        _, train_acc, train_loss, train_summary = self.sess.run(
            [self.optim, self.acc, self.loss, self.summary_op],
            feed_dict=feed_dict)

      # Save every 10 iterations
      if j % 10 == 0:
        self.save(self.global_step)

      if j % 1 == 0:
        # Report training
        self.logger.info("Epoch: " + str(j) + " has train loss: " +
                         str(float(train_loss)) + " and train accuracy: " +
                         str(float(train_acc)))
        self.train_writer.add_summary(train_summary, global_step=j)

        # Report test
        feed_dict = {
            self.x: test_X[:self.config.BATCH_SIZE],
            self.target: test_Y[:self.config.BATCH_SIZE]
        }
        test_acc, test_loss, test_summary = self.sess.run(
            [self.acc, self.loss, self.summary_op],
            feed_dict=feed_dict)
        self.logger.info("Epoch: " + str(j) + " has test loss: " +
                         str(float(test_loss)) + " and test accuracy: " +
                         str(float(test_acc)))
        self.test_writer.add_summary(test_summary, global_step=j)

    self.logger.info('Model finished training!')

  def predict(self, X):
    '''
    Predict classifications for new inputs.
    Args:
      X (np.arr): featured data. Assuming len(X) > batch size.
                  Note that left over data from batching are
                  NOT calculated. So pad batches beforehand.
    Returns:
      predictions (list of np.arr): flat list of predictions.
    '''

    self.logger.info('Starting model predictions...')
    predictions = []
    for i in range(0, len(X) + 1 - self.config.BATCH_SIZE,
                   self.config.BATCH_SIZE):
      feed_dict = {
          self.x: X[i:i + self.config.BATCH_SIZE]
      }
      predictions += list(self.sess.run([self.prediction],
                          feed_dict=feed_dict)[0])
    self.logger.info('Model finished predicting!')
    return np.array(predictions)

  def shuffle_and_partition(self, X, Y, n_test, n_val):
    '''
    Shuffle and partition input data.
    Args:
      - X (np.array): X data
      - Y (np.array): target to be shuffled in sync
      - n_test (int): number of test points
      - n_val (int): number of validation points
    Return:
      - result:
        {"train": {"X": np.arr, "Y": np.arr},
        "test": {"X": np.arr, "Y": np.arr},
        "val": {"X": np.arr, "Y": np.arr}}.
    '''

    n_train = X.shape[0] - n_test - n_val
    self.logger.debug('Shuffling and partitioning data...')

    self.logger.debug('Shuffling X, Y in sync...')
    p = np.random.permutation(X.shape[0])
    shuffled_X, shuffled_Y = X[p], Y[p]
    del(p)
    del(X)
    del(Y)
    self.logger.debug('Finished shuffling dataset.')

    # Structured as: [train, test, val]
    self.logger.debug('Partitioning with training size: ' + str(n_train) +
                      ' test size: ' + str(n_test) + ' and val size: ' +
                      str(n_val) + '...')
    self.logger.debug('First partitioning X...')
    train_X = shuffled_X[:n_train]
    test_X = shuffled_X[n_train:(n_train + n_test)]
    val_X = shuffled_X[(n_train + n_test):]
    del(shuffled_X)

    self.logger.debug('Finished partitioning X. Now partitioning Y...')
    train_Y = shuffled_Y[:n_train]
    test_Y = shuffled_Y[n_train:(n_train + n_test)]
    val_Y = shuffled_Y[(n_train + n_test):]
    del(shuffled_Y)
    self.logger.debug('Finished partitioning Y.')

    self.logger.info('Finished shuffling and partitioning.')

    uniques, counts = np.unique(np.argmax(train_Y, 1), return_counts=True)
    for i in range(len(uniques)):
      self.logger.debug(
          'Class: ' + str(uniques[i]) + '; count: ' + str(counts[i]))

    return {"train": {"X": train_X, "Y": train_Y},
            "test": {"X": test_X, "Y": test_Y},
            "val": {"X": val_X, "Y": val_Y}}

