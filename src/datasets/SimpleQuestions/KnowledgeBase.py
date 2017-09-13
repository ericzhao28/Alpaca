from neo4j.v1 import GraphDatabase, basic_auth


class KnowledgeBase():
  def __init__(self):
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "neo4j"))
    self.session = driver.session()
    self.entity_names = []

  def import_edge(self, edge):
    self.add_entity(edge[0])
    self.add_entity(edge[2])
    self.add_relationship(edge[2], edge[0], edge[1])

  def add_entity(self, name, key=None, value=None):
    if key is not None:
      self.session.run("MERGE (n:Entity {name:'%s'})" % (name))
    else:
      self.session.run("MERGE (n:Entity {name:'%s', %s:'%s'})" % (name, key, value))
    if name not in self.entity_names:
      self.entity_names.append(name)

  def add_relationship(self, parent_name, child_name, relationship):
    query = """
      MATCH (a:Entity), (b:Entity)
      WHERE a.name = '%s' AND b.name = '%s'
      CREATE (a)-[r:'%s']->(b)
      RETURN r
    """ % (child_name, parent_name, relationship)
    self.session.run(query)

  def get_entity(self, entity):
    return self.session.run("MATCH (n:Entity {name:'%s'}) RETURN n" % entity)

  def get_all_relations(self):
    return self.session.run("MATCH (a)-[r]->(b) RETURN r")

  def update_entity(self, name, key, value, create_if_none=False):
    if create_if_none:
      if not self.get_entity(name):
        self.add_entity(name)
    query = """
      MATCH (a:Entity)
      WHERE a.name = '%s'
      SET a.%s = '%s'
      RETURN a
    """ % (name, key, value)
    self.session.run(query)

  def get_parents(self, entity, relationship=None):
    if relationship is None:
      query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.name = '%s'
        RETURN b
      """ % (entity)
    else:
      query = """
        MATCH (a:Entity)-[r:%s]->(b:Entity)
        WHERE a.name = '%s'
        RETURN b
      """ % (relationship, entity)
    for parent in self.session.run(query):
      yield parent

  def get_childs(self, entity, relationship=None):
    if relationship is None:
      query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE b.name = '%s'
        RETURN a
      """ % (entity)
    else:
      query = """
        MATCH (a:Entity)-[r:%s]->(b:Entity)
        WHERE b.name = '%s'
        RETURN a
      """ % relationship, entity
    for child in self.session.run(query):
      yield child

  def wipe(self):
    query = """
      MATCH (n)
      DETACH DELETE n
    """
    self.session.run(query)

  def disconnect(self):
    self.session.close()

