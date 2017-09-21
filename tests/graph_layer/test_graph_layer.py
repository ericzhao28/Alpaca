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

  assert(len(graph.get_all_relations()) == 1)
  assert(len(graph.entity_names) == 2)
  assert(graph.get_parents("chocolate") == "hugs")
  assert(graph.get_parents("chocolate", "ideal") == "hugs")
  assert(graph.get_parents("chocolate", "ideal2") == False)
  assert(graph.get_childs("hugs") == "chocolate")
  assert(graph.get_childs("hugs", "ideal") == "chocolate")
  assert(graph.get_childs("hugs", "ideal2") == False)
  assert(graph.get_entity("chocolate2") == False)

  graph.import_edge("hugs2", "ideal2", "chocolate2")
  graph.update_entity("chocolate2", "test", "True")
  graph.update_entity("hugs2", "test", "True")
  assert(graph.get_entity("chocolate2")['test'] == "True")
  assert(graph.get_entity("hugs2")['test'] == "True")

  assert(len(graph.get_all_relations()) == 2)
  assert(len(graph.entity_names) == 4)

  graph.wipe_tests()

