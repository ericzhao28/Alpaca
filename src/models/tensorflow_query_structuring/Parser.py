from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from . import config, Layered_Model
import tensorflow as tf
import pickle


class Parser(Layered_Model.Layered_Model):
  def __init__(self, sess):
    Layered_Model.__init__(sess)

    with open(config.DATA_DIR + config.VOCAB_SAVE, 'rb') as f:
      self.inv_vocab = pickle.load(f)
    self.vocab = {}
    for i, word in enumerate(self.inv_vocab):
      self.vocab[word] = i
    model = KeyedVectors.load_word2vec_format(config.DATA_DIR + config.EMBEDDING_SAVE, binary=True)
    self.embeddings = np.zeroes((len(self.inv_vocab), config.NUMBERS['emb_dim']))
    for k, v in self.vocab.items():
      self.embeddings[v] = model[k]

  def build_model(self):
    self.embedded_x = tf.nn.embedding_lookup(self.embeddings, self.x)

    encoder_config = {
        'n_seqs': config.BATCH_SIZE,
        'seq_len': config.NUMBERS['seq_len'],
        'n_features': config.NUMBERS['emb_dim'],
        'n_gru_hidden': config.NUMBERS['gru_h'],
        'n_attention_hidden': config.NUMBERS['attention_h'],
        'n_dense_hidden': config.NUMBERS['state_h']
    }

    encoded_state, encoded_seq = self._attention_encoder_layer(self.embedded_x, "encoder", encoder_config)

    # Get entity predictions
    entity_predictor_config = {
        'n_batches': config.BATCH_SIZE * config.NUMBERS['seq_len'],
        'n_input': config.NUMBERS['gru_h'],
        'n_classes': config.NUMBERS['n_entities']
    }
    flat_encoded_seq = tf.reshape(encoded_seq, (config.BATCH_SIZE * config.NUMBERS['seq_len'], config.NUMBERS['gru_h']))
    flat_entity_prediction = self._prediction_layer(flat_encoded_seq, 'entity_predictor', entity_predictor_config)
    self.entity_prediction = tf.reshape(flat_entity_prediction, (config.BATCH_SIZE, config.NUMBERS['seq_len'], config.NUMBERS['n_entities']))

    # Get entity predictions
    relationship_predictor_config = {
        'n_batches': config.BATCH_SIZE,
        'n_input': config.NUMBERS['state_h'],
        'n_classes': config.NUMBERS['n_relationships']
    }
    self.relationship_prediction = self._prediction_layer(encoded_state, 'relationship_predictor', relationship_predictor_config)

    # Get loss and optimizer
    entity_loss, entity_acc = self._define_optimization_vars(self.entity_target, self.entity_prediction, config.LOSS_WEIGHTING)
    relationship_loss, relationship_acc = self._define_optimization_vars(self.relationship_target, self.relationship_prediction, config.LOSS_WEIGHTING)
    self.acc = (entity_acc + relationship_acc) / tf.constant(2)
    self.loss = tf.add(entity_loss, relationship_loss)
    optimizer = tf.train.AdamOptimizer()
    self.optim = optimizer.minimize(self.loss, var_list=tf.trainable_variables(), global_step=self.global_step)
    self.summary_op = self._summaries()

    return self

  def predict_entities(self):
    return self.predict(self.entity_prediction)

  def predict_relationship(self):
    return self.predict(self.relationship_prediction)

