from ...src.models import Parser
import tensorflow as tf


def test_initialize():
  with tf.Session() as sess:
    model = Parse(sess)
    model.initialize()
    model.save()
    model2 = Parse(sess)
    model2.restore()

