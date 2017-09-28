from ..model_base import Base, SequenceLayers
import tensorflow as tf


class FlatModel(Base, SequenceLayers):
  '''
  Model for predicting on flat data.
  Model-specific config requirements:
    BATCH_SIZE
    N_FEATURES
    N_CLASSES
  '''

  def __init__(self, sess, config, logger):
    logger.info('Instantiated flat model')
    Base.__init__(self, sess, config, logger)

  def build_model(self):
    '''
    Build the flat model.
    '''

    self.logger.info('Building model...')
    config = self.config

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_FEATURES))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    encoder_one_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.N_FEATURES,
        'n_hidden': config.LAYERS['h_one_encoder'],
        'n_output': config.LAYERS['h_one_encoded']
    }
    self.encoded_one = self._dense_layer(
        self.x,
        'encoder_one',
        encoder_one_config)

    encoder_two_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.LAYERS['h_one_encoded'],
        'n_hidden': config.LAYERS['h_two_encoder'],
        'n_output': config.LAYERS['h_two_encoded']
    }
    self.encoded_two = self._dense_layer(
        self.encoded_one,
        'encoder_two',
        encoder_two_config)

    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.LAYERS['h_two_encoded'],
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        self.encoded_two,
        'predictor',
        predictor_config)

    self.loss, self.acc = self._define_optimization_vars(
        self.target,
        self.prediction,
        None)
    optimizer = tf.train.AdamOptimizer()
    self.optim = optimizer.minimize(
        self.loss,
        var_list=tf.trainable_variables(),
        global_step=self.global_step)
    self.summary_op = self._summaries()

    self.logger.info('Model built.')

    return self

