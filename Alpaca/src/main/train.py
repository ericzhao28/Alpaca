from ..models import EntModel, RelModel, TFIDF
from ..graph.load import create_graph
from . import rel_config, ent_config
from .logger import train_logger

from ...datasets.SimpleQuestions import load

from neo4j.v1 import GraphDatabase
import tensorflow as tf


def train():
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo4j_sess:
    with tf.Session() as tf_sess:
      # Create graph
      graph = create_graph(neo4j_sess, load())

      # Train tf_idf
      tf_idf_model = TFIDF()
      tf_idf_model.train([x.split(" ") for x in graph.entity_names])

      # Train ent_model
      ent_model = EntModel(tf_sess, ent_config, train_logger)
      X, Y = load()
      shuffled_dataset = ent_model.shuffle_and_partition(X, Y, 20, 20)
      del(X)
      del(Y)
      ent_model.initialize()
      ent_model.train(
          shuffled_dataset['train']['X'],
          shuffled_dataset['train']['Y'],
          shuffled_dataset['test']['X'],
          shuffled_dataset['test']['Y']
      )
      ent_model.save()
      del(shuffled_dataset)

      # Train rel model
      rel_model = RelModel(tf_sess, rel_config, train_logger)
      X, Y = load()
      shuffled_dataset = rel_model.shuffle_and_partition(X, Y, 20, 20)
      del(X)
      del(Y)
      rel_model.initialize()
      rel_model.train(
          shuffled_dataset['train']['X'],
          shuffled_dataset['train']['Y'],
          shuffled_dataset['test']['X'],
          shuffled_dataset['test']['Y']
      )
      rel_model.save()
      del(shuffled_dataset)


if __name__ == "__main__":
  train()

