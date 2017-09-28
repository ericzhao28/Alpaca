import logging
from ...logs import setup_logger


graph_logger = setup_logger('GraphLayer', 'graph/', logging.DEBUG)
load_logger = setup_logger('GraphLoad', 'graph/', logging.DEBUG)

