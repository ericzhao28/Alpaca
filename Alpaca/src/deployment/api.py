from flask import request, jsonify
from flask_api import FlaskAPI
from flask_cors import CORS
from ..main import load
from .logger import api_logger
import tensorflow as tf

app = FlaskAPI(__name__)
cors = CORS(app)


with tf.Session() as sess:
  execute_func = None

  @app.route("/demo", methods=['POST'])
  def get_demo():
    """
    Simple synchronous endpoint for demo.

    Make query to 0.0.0.0:5000/query with
    POST protocol, payload: {"query": "hi"}.
    Response: {"response": "hello"}.
    """
    text = str(request.data.get('query', ''))
    api_logger.debug('Received query: ' + text)
    response = execute_func(text)
    api_logger.debug('Responding with: ' + response)
    return jsonify({'response': response})

  if __name__ == "__main__":
    execute_func = load.initialize_func(sess, api_logger)
    api_logger.info('Starting deployment API on 0.0.0.0...')
    app.run(host='0.0.0.0')

