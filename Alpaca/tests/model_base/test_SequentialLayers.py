from ...src.model_base import SequenceLayers
import tensorflow as tf


def test_word2vec():
  pass


def test_vanilla():
  raw = tf.placeholder(tf.float32, (3, 11, 7))

  encoder_config = {
      'n_batches': 3,
      'n_steps': 11,
      'n_features': 7,
      'h_gru': 13,
      'h_att': 17,
      'h_dense': 19
  }
  encoded_state, encoded_seq = SequenceLayers()._encoder_layer(
      raw, "encoder", encoder_config)

  assert(encoded_state.shape == (3, 19))
  assert(encoded_state.dtype == tf.float32)
  assert(encoded_seq.shape == (3, 11, 2 * 13))
  assert(encoded_seq.dtype == tf.float32)


def test_attention():
  raw = tf.placeholder(tf.float32, (3, 11, 7))

  encoder_config = {
      'n_batches': 3,
      'n_steps': 11,
      'n_features': 7,
      'h_gru': 13,
      'h_att': 17,
      'h_dense': 19
  }
  encoded_state, encoded_seq = SequenceLayers()._attention_encoder_layer(
      raw, "attention", encoder_config)

  assert(encoded_state.shape == (3, 19))
  assert(encoded_state.dtype == tf.float32)
  assert(encoded_seq.shape == (3, 11, 2 * 13))
  assert(encoded_seq.dtype == tf.float32)

