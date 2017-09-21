from ...src.graph_layer import Graph_Layer


def test_graph():
  graph = Graph_Layer()
  graph.wipe_tests()
  graph.add_entity("chocolate")
  graph.update_entity("chocolate", "test", "True")
  graph.add_entity("hugs", "test", "True")
  assert(graph.get_entity("chocolate")['test'] == "True")
  assert(graph.get_entity("hugs")['test'] == "True")
  graph.add_relationship("hugs", "chocolate", "ideal")

  graph.import_edge("hugs2", "ideal2", "chocolate2")
  graph.update_entity("chocolate2", "test", "True")
  graph.update_entity("hugs2", "test", "True")
  assert(graph.get_entity("chocolate2")['test'] == "True")
  assert(graph.get_entity("hugs2")['test'] == "True")

  for rel in graph.get_all_relations():
    assert(rel == "")

  graph.wipe_tests()

