from flask import request, jsonify
from flask.ext.api import FlaskAPI
from flask_cors import CORS
from . import execute

app = FlaskAPI(__name__)
cors = CORS(app)

execute_func = execute.initialize()


@app.route("/demo", methods=['POST'])
def get_chat_demo():
  """For a text query, pipe it through the gate and return the best answer."""
  text = str(request.data.get('query', ''))
  response = execute_func(text)
  return jsonify({'response': response})


if __name__ == "__main__":
  app.run(host='0.0.0.0')

