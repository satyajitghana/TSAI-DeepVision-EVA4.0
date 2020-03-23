import logging
import sys

LOG_LEVEL = logging.INFO


def setup_logger(name):
    logger = logging.getLogger(f'sodium.{name}')
    logger.setLevel(LOG_LEVEL)  # set the logging level

    # logging format
    logger_format = logging.Formatter(
        '[ %(asctime)s - %(name)s ] %(levelname)s: %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logger_format)
    logger.addHandler(stream_handler)

    return logger  # return the logger
