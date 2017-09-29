from ..model_base import Base, SequenceLayers
import tensorflow as tf


class DoubleSeqModel(Base, SequenceLayers):
  '''
  Model for predicting on double sequence data.
  Model-specific config requirements:
    BATCH_SIZE
    N_SEQS
    N_STEPS
    N_FEATURES
    N_CLASSES
    SEQ_ENCODED_DIM
    ENCODED_DIM
    LAYERS: h_seq_gru, h_seq_att, h_step_gru, h_step_att
  '''

  def __init__(self, sess, config, logger):
    logger.info('Instantiated double sequential model')
    Base.__init__(self, sess, config, logger)

  def build_model(self):
    '''
    Build the double sequential model.
    '''

    self.logger.info('Building model...')
    config = self.config

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_SEQS, config.N_STEPS,
                     config.N_FEATURES))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    step_encoder_config = {
        'n_batches': config.BATCH_SIZE * config.N_SEQS,
        'n_steps': config.N_STEPS,
        'n_features': config.N_FEATURES,
        'h_gru': config.LAYERS['h_step_gru'],
        'h_att': config.LAYERS['h_step_att'],
        'h_dense': config.SEQ_ENCODED_DIM
    }
    step_encoded_state, _ = self._attention_encoder_layer(
        tf.reshape(self.x, (
            config.BATCH_SIZE * config.N_SEQS,
            config.N_STEPS,
            config.N_FEATURES
        )),
        "step_encoder", step_encoder_config
    )

    seq_encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_SEQS,
        'n_features': config.SEQ_ENCODED_DIM,
        'h_gru': config.LAYERS['h_seq_gru'],
        'h_att': config.LAYERS['h_seq_att'],
        'h_dense': config.ENCODED_DIM
    }
    encoded_state, encoded_seq = self._attention_encoder_layer(
        tf.reshape(step_encoded_state, (
            config.BATCH_SIZE,
            config.N_SEQS,
            config.SEQ_ENCODED_DIM
        )),
        "seq_encoder", seq_encoder_config
    )

    predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.ENCODED_DIM,
        'n_classes': config.N_CLASSES
    }
    self.prediction = self._prediction_layer(
        encoded_state,
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

