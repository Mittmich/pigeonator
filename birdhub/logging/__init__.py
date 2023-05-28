import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from .loggers import EventLogger

# setup up logging

logging.setLoggerClass(EventLogger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# add logfile handler
log_dir = os.getenv("PGN_LOGDIR", None)
if log_dir is None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "birdhub.log")
fh = RotatingFileHandler(log_file, maxBytes=2000, backupCount=10)
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter(EventLogger.LOG_FORMAT, "%Y-%m-%dT%H:%M:%S")
handler.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(fh)