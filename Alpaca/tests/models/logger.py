import logging
from ...logs import setup_logger


double_seq_logger = setup_logger('double_seq', 'test_logs/', logging.DEBUG)
sequential_logger = setup_logger('sequential', 'test_logs/', logging.DEBUG)
flat_logger = setup_logger('flat', 'test_logs/', logging.DEBUG)

