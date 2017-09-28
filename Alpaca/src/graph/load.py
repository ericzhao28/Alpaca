from .graph import Graph
from .logger import load_logger


def create_graph(sess, edges, test=False):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''

  load_logger.info('Creating graph with ' + str(len(edges)) + ' edges.')
  cls = "main" if test is False else "test"

  graph = Graph(sess)
  for parent, rel, child in edges:
    load_logger.debug('Autocreating edge: (' + parent + ') - [' +
                      rel + '] -> (' + child + ').')
    parent_nd = graph.build_node(name=parent, cls=cls, ind='a')
    child_nd = graph.build_node(name=child, cls=cls, ind='b')
    edge = graph.build_edge(rel=rel)

    graph.add_entity(parent_nd)
    graph.add_entity(child_nd)
    graph.add_relationship(parent_nd, child_nd, edge)

  load_logger.info('Graph with ' + str(len(edges)) + ' edges is created.')
  return graph


def load_graph(sess):
  '''
  Generate empty graph without loading in additional data,
  assuming server-side Neo4j graph already exists.
  '''

  load_logger.info('Loading graph without automatic creation.')
  return Graph(sess)

