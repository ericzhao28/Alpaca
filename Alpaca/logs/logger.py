import logging
from . import config
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, directory, level=logging.DEBUG):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(config.LOG_ROOT + directory + name + ".log")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

