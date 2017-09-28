from .driver_medium import DriverMedium
from .logger import graph_logger


class Graph(DriverMedium):
  '''
  Graph class for holding basic graph logic.
  '''

  def __init__(self, sess):
    self.sess = sess
    graph_logger.info('Creating new Neo4j graph')

  def disconnect(self):
    '''
    Terminate neo4j session.
    '''

    graph_logger.info('Terminating Neo4j session...')
    self.sess.close()
    graph_logger.info('Neo4j session terminated.')

  @property
  def entity_names(self):
    '''
    Return names of all entities.
    '''

    for record in self.get_entity("(n)"):
      yield(record[0].properties['name'])

  def get_entity(self, node):
    '''
    Get an entity.
    '''

    query = "MATCH %s RETURN n" % node
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

  def get_relationship(self, parent_nd, child_nd, edge):
    '''
    Get a relationship from the graph.
    '''

    query = "MATCH %s-%s->%s RETURN r" % (parent_nd, edge, child_nd)
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

  def add_entity(self, node):
    '''
    Add an entity to the graph.
    '''

    query = "MERGE %s" % node
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

  def add_relationship(self, parent_nd, child_nd, edge):
    '''
    Add a relationship to the graph.
    '''

    query = "MERGE %s-%s->%s" % (parent_nd, edge, child_nd)
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

  def wipe(self):
    '''
    Wipe all nodes from graph.
    '''
    query = "MATCH (n) DETACH DELETE n"
    graph_logger.info('Wiping all nodes...')
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

  def wipe_tests(self):
    '''
    Wipe all test nodes from graph.
    '''
    query = "MATCH (n:test) DETACH DELETE n"
    graph_logger.info('Wiping all test nodes...')
    graph_logger.debug("Cypher exec: " + query)
    return self.sess.run(query)

