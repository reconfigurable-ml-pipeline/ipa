import logging
import inspect
from .constants import LOGGING_LEVEL, LOG_TO

if LOGGING_LEVEL == "debug":
    level = logging.DEBUG
elif LOGGING_LEVEL == "info":
    level = logging.INFO
elif LOGGING_LEVEL == "warning":
    level = logging.WARNING
elif LOGGING_LEVEL == "error":
    level = logging.ERROR
else:
    raise ValueError("Unknown logging level " + LOGGING_LEVEL)

log_format = " %(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"

if LOG_TO == "print":
    logging.basicConfig(format=log_format, datefmt="%Y-%m-%d:%H:%M:%S", level=level)
else:
    logging.basicConfig(
        format=log_format, datefmt=log_format, filename=LOG_TO, level=level
    )


def debug(msg):
    logging.debug(msg)


def info(msg):
    logging.info(msg)


def warn(msg):
    logging.warning(msg)


def error(msg):
    logging.error(msg)


def exception(msg, *args, **kwargs):
    logging.exception(msg, *args, **kwargs)
