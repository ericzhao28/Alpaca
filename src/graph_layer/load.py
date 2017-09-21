from . import Graph_Layer
from ...utils import cleaning


def create_graph(path):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''
  graph = Graph_Layer()
  with open(path, 'r') as f:
    for line in f.readlines():
      assert(len(line.split(" ")) == 3)
      graph.import_edge([cleaning.clean_name(x) for x in line.split(" ")])
  return graph


def load_graph():
  return Graph_Layer()

