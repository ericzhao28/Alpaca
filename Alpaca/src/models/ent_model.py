from ..model_base import Base, SequenceLayers
import tensorflow as tf


class EntModel(Base, SequenceLayers):
  '''
  Model for predicting entities of query.
  Model-specific config requirements:
    BATCH_SIZE
    N_STEPS
    EMB_DIM
    EMB_PATH
    N_CLASSES
    LAYERS: h_seq_int, h_seq_output, h_ent_int, h_ent_output
  '''

  def __init__(self, sess, config, logger):
    logger.info('Instantiated sequential model')
    Base.__init__(self, sess, config, logger)

  def build_model(self):
    '''
    Build the sequential model.
    '''

    self.logger.info('Building model...')
    config = self.config

    self.x = tf.placeholder(
        tf.float32, (config.BATCH_SIZE, config.N_STEPS))
    self.target = tf.placeholder(tf.float32,
                                 (config.BATCH_SIZE, config.N_CLASSES))

    embedder_config = {
        'emb_dim': config.EMB_DIM,
        'emb_path': config.EMB_PATH
    }
    embedded_x = self._embedding_layer(
        self.x, "embedder", embedder_config)

    seq_encoder_config = {
        'n_batches': config.BATCH_SIZE,
        'n_steps': config.N_STEPS,
        'n_features': config.EMB_DIM,
        'h_gru': config.LAYERS['h_seq_int'],
        'h_dense': config.LAYERS['h_seq_output']
    }
    _, encoded_seq = self._encoder_layer(
        embedded_x, "seq_encoder", seq_encoder_config)

    ent_encoder_config = {
        'n_batches': config.BATCH_SIZE * config.N_STEPS,
        'n_input': config.LAYERS['h_seq_output'] * 2,
        'n_hidden': config.LAYERS['h_ent_int'],
        'n_output': config.LAYERS['h_ent_output']
    }
    encoded_entities = self._dense_layer(
        encoded_seq,
        'ent_encoder',
        ent_encoder_config
    )

    predictor_config = {
        'n_batches': config.BATCH_SIZE * config.N_STEPS,
        'n_input': config.LAYERS['h_ent_output'],
        'n_classes': config.N_CLASSES
    }
    self.prediction = tf.reshape(
        self._prediction_layer(
            encoded_entities,
            'predictor',
            predictor_config
        ), (
            config.BATCH_SIZE,
            config.N_STEPS,
            config.N_CLASSES
        )
    )

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

