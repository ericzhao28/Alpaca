from ...src.graph import DriverMedium
from collections import OrderedDict


def test_build_node():
  driver = DriverMedium()
  assert("(n)" == driver.build_node())
  assert("(a)" == driver.build_node(ind="a"))
  assert("(a:test)" == driver.build_node(cls="test", ind="a"))
  assert("(n:test)" == driver.build_node(cls="test", ind="n"))
  assert("(n {name:'yay'})" == driver.build_node(name="yay"))

  d = OrderedDict(sorted({'no': 'what', 'woo': 'hello'}.items(),
                         key=lambda t: t[0]))
  assert("(n {no:'what', woo:'hello', name:'yay'})" ==
         driver.build_node(name="yay", properties=d))
  d = OrderedDict(sorted({'no': 'what', 'woo': 'hello'}.items(),
                         key=lambda t: t[0]))
  assert("(n {no:'what', woo:'hello'})" == driver.build_node(properties=d))
  d = OrderedDict(sorted({'no': 'what', 'woo': 'hello'}.items(),
                         key=lambda t: t[0]))
  assert("(n:test {no:'what', woo:'hello', name:'yay'})" ==
         driver.build_node(cls="test", name="yay", properties=d))
  d = OrderedDict(sorted({'no': 'what', 'woo': 'hello'}.items(),
                         key=lambda t: t[0]))
  assert("(n:test {no:'what', woo:'hello'})" ==
         driver.build_node(cls="test", properties=d))
  d = OrderedDict(sorted({'no': 'what', 'woo': 'hello'}.items(),
                         key=lambda t: t[0]))
  assert("(a:test {no:'what', woo:'hello'})" ==
         driver.build_node(cls="test", properties=d, ind="a"))


def test_build_edge():
  driver = DriverMedium()
  assert("[r]" == driver.build_edge())
  assert("[r:friends]" == driver.build_edge("friends"))
  assert("[b]" == driver.build_edge(ind="b"))
  assert("[b:friends]" == driver.build_edge("friends", "b"))


def test_entity_to_node():
  pass

