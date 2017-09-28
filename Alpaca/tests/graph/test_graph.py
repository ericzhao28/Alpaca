from ...src.graph import Graph, load
from neo4j.v1 import GraphDatabase


def does_child1_graph_exists(graph):
  result = graph.get_entity(graph.build_node(name="child1")).single()
  assert(result[0])
  assert(result[0].properties == {"name": "child1"})
  assert(result[0].labels == {"test"})
  return True


def does_child1parent2_exists(graph):
  result = graph.get_relationship(
      graph.build_node(name="parent2", ind='a'),
      graph.build_node(name="child1", ind='b'),
      graph.build_edge(rel="mom")).single()
  assert(result[0].type == "mom")
  return result


def does_child1parent2_not_exists(graph):
  result = graph.get_relationship(
      graph.build_node(name="parent2", ind='a'),
      graph.build_node(name="child1", ind='b'),
      graph.build_edge(rel="mom")).single()
  assert(result is None)
  return not result


def test_graph():
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    graph = Graph(neo_sess)
    graph.wipe_tests()
    edges = [("parent1", "dad", "child1"),
             ("parent2", "mom", "child1"),
             ("parent1", "dad", "child2"),
             ("parent2", "mom", "child2"),
             ("parent1", "uncle", "child3"),
             ("parent2", "aunt", "child3")]

    for parent, rel, child in edges:
      parent_nd = graph.build_node(name=parent, cls="test", ind='a')
      child_nd = graph.build_node(name=child, cls="test", ind='b')
      edge = graph.build_edge(rel=rel)
      graph.add_entity(parent_nd)
      graph.add_entity(child_nd)
      graph.add_relationship(parent_nd, child_nd, edge)

    assert(set(graph.entity_names) == {"child1", "child2",
                                       "child3", "parent1",
                                       "parent2"})

    assert(does_child1_graph_exists(graph))
    assert(does_child1parent2_exists(graph))
    assert(not graph.get_entity(graph.build_node(name="sdf")).single())

    graph.wipe_tests()

    assert(not graph.get_entity(graph.build_node(name="child1")).single())
    assert(list(graph.entity_names) == [])


def test_load():
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    edges = [("parent1", "dad", "child1"),
             ("parent2", "mom", "child1"),
             ("parent1", "dad", "child2"),
             ("parent2", "mom", "child2"),
             ("parent1", "uncle", "child3"),
             ("parent2", "aunt", "child3")]
    graph = load.create_graph(neo_sess, edges, test=True)
    assert(set(graph.entity_names) == {"child1", "child2",
                                       "child3", "parent1",
                                       "parent2"})

    assert(does_child1_graph_exists(graph))
    assert(does_child1parent2_exists(graph))
    assert(not graph.get_entity(graph.build_node(name="sdf")).single())

  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    graph = load.load_graph(neo_sess)

    assert(set(graph.entity_names) == {"child1", "child2",
                                       "child3", "parent1",
                                       "parent2"})

    assert(does_child1_graph_exists(graph))
    assert(does_child1parent2_exists(graph))
    assert(not graph.get_entity(graph.build_node(name="sdf")).single())

    graph.wipe_tests()

    assert(not graph.get_entity(graph.build_node(name="child1")).single())
    assert(does_child1parent2_not_exists(graph))
    assert(list(graph.entity_names) == [])

