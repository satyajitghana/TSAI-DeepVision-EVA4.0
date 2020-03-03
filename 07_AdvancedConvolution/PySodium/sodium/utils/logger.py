import logging

LOG_LEVEL = logging.INFO


def setup_logger(name):
    logger = logging.getLogger(f'sodium.{name}')
    logger.setLevel(LOG_LEVEL)  # set the logging level
    return logger  # return the logger
