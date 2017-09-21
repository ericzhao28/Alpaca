from ...src.graph_layer import load
from . import config


def test_create_graph():
  graph = load.create_graph(config.DATA_DIR + config.DUMP_SAVE)
  assert(graph.get_all_relations() == ["is", "not"])
  assert(graph.entity_names() == ["chocolate", "hugs", "people", "good", "bad"])
  assert(graph.get_childs("chocolate") == ["good", "bad"])
  assert(graph.get_childs("chocolate", "not") == ["bad"])
  graph.update_entity("chocolate", "test", "True")
  graph.update_entity("hugs", "test", "True")
  graph.update_entity("people", "test", "True")
  graph.update_entity("good", "test", "True")
  graph.update_entity("bad", "test", "True")
  graph.wipe_tests()


def test_load_graph():
  load.create_graph(config.DATA_DIR + config.DUMP_SAVE)
  graph = load.load_graph()
  assert(graph.get_all_relations() == ["is", "not"])
  assert(graph.entity_names() == ["chocolate", "hugs", "people", "good", "bad"])
  assert(graph.get_childs("chocolate") == ["good", "bad"])
  assert(graph.get_childs("chocolate", "not") == ["bad"])
  graph.update_entity("chocolate", "test", "True")
  graph.update_entity("hugs", "test", "True")
  graph.update_entity("people", "test", "True")
  graph.update_entity("good", "test", "True")
  graph.update_entity("bad", "test", "True")
  graph.wipe_tests()

