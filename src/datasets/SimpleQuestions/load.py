from . import KnowledgeBase
from ...utils import cleaning


def create_graph():
  new_graph = KnowledgeBase()
  with open('freebase.txt', 'r') as f:
    for line in f.readlines():
      assert(len(line.split()) == 3)
      new_graph.import_edge([cleaning.clean_name(x) for x in line])
  return new_graph


def load_graph():
  return KnowledgeBase()

