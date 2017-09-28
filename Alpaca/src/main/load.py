from ..models import SequentialModel
from . import config


def initialize_func(sess, logger):
  '''
  Initialize main function to predict using model.
  '''
  logger.debug('Starting Tensorflow session.')
  model = SequentialModel(sess, config, logger)
  logger.debug('Model instantiated.')
  model.initialize()
  logger.debug('Model initialized.')
  model.restore()
  logger.debug('Model restored.')

  def function(query):
    result = model.predict([query])[0]
    logger.debug('Model predicted: ' + str(result))
    return result

  return function

