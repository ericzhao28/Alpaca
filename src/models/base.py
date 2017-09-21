import numpy as np
import tensorflow as tf
from . import config


class Base_Model():

  def __init__(self, sess):
    tf.set_random_seed(4)
    self.sess = sess
    self.saver = None
    self.model_name = "default.model"
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

  def initialize(self):
    '''
    Initialize models: builds model, loads data, initializes variables
    '''
    filename_queue = tf.train.string_input_producer([self.data_config.DATA_DIR + self.data_config.TF_SAVE])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    self.load_pipeline(serialized)
    self.build_model()
    self.writer = tf.summary.FileWriter(self.data_config.DATA_DIR + 'graphs', self.sess.graph)
    self.var_init = tf.global_variables_initializer()
    self.var_init.run()

  def save(self, global_step=None):
    '''
    Save the current variables in graph.
    Optional option to save for global_step (used in Train)
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())
    if global_step is None:
      self.saver.save(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name)
    else:
      self.saver.save(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name, global_step=self.global_step)

  def restore(self, resume=False):
    '''
    Load saved variable values into graph
    '''
    if self.saver is None:
      self.saver = tf.train.Saver(tf.global_variables())

    if resume:
      ckpt = tf.train.get_checkpoint_state(self.data_config.DATA_DIR + 'checkpoints/checkpoint')
      if ckpt and ckpt.model_checkpoint_path:
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      return

    self.saver.restore(self.sess, self.data_config.DATA_DIR + 'checkpoints/' + self.model_name)

  def load_pipeline(self, serialized):
    '''
    Load stream of data and batches input
    '''
    features = tf.parse_single_example(
        serialized,
        features={
            'features': tf.FixedLenFeature([], tf.string),
            'entity_label': tf.FixedLenFeature([], tf.string),
            'relationship_label': tf.FixedLenFeature([], tf.string),
        }
    )
    features = tf.reshape(tf.decode_raw(features['features'], tf.string), config.NUMBERS['seq_len'])
    entity_label = tf.reshape(tf.decode_raw(features['entity_label'], tf.float32), config.NUMBERS['n_entities'])
    relationship_label = tf.reshape(tf.decode_raw(features['relationship_label'], tf.float32), config.NUMBERS['n_relationship'])

    self.x, self.entity_target, self.relationship_target = tf.train.shuffle_batch([features, entity_label, relationship_label], batch_size=config.BATCH_SIZE, capacity=500, min_after_dequeue=100)

  def train(self):
    '''
    Run model training. Model must have been initialized.
    '''

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    try:
      i = 0
      while not coord.should_stop():
        raw_x, raw_entity_y, raw_relationship_y, _, acc, loss, summary = self.sess.run([self.x, self.entity_target, self.relationship_target, self.optim, self.acc, self.loss, self.summary_op])
        i += 1
        print("Epoch:", i, "has loss:", loss, "and accuracy:", acc)
        self.writer.add_summary(summary, global_step=self.global_step)
    finally:
      coord.request_stop()
      coord.join(threads)

  def predict(self, to_predict):
    '''
    Predict classifications for new inputs
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
    predictions = []
    try:
      while not coord.should_stop():
        raw_x, prediction = self.sess.run([self.x, to_predict])
        predictions += np.argmax(prediction, axis=1)
    finally:
      coord.request_stop()
      coord.join(threads)
    return predictions

