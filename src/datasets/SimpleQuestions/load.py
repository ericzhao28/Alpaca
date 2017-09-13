from . import KnowledgeBase
from ...utils import cleaning


def create_graph():
  new_graph = KnowledgeBase()
  with open('freebase.txt', 'r') as f:
    for line in f.readlines():
      assert(len(line.split()) == 3)
      new_graph.add_entity(cleaning.clean_name(line[0]))
      new_graph.add_entity(cleaning.clean_name(line[2]))
      new_graph.add_relationship(cleaning.clean_name(line[2]), cleaning.clean_name(line[0]), cleaning.clean_name(line[1]))
  return new_graph


def load_graph():
  return KnowledgeBase()
