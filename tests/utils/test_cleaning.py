from ...src.utils import cleaning


def test_clean_name():
  assert("woo" == cleaning.clean_name("  woo "))
  assert("woo" == cleaning.clean_name("\twoo "))
  assert("woo" == cleaning.clean_name("\tWoo "))
