import logging
import os
import sys

from .. import config


def setup_logger():
    try:
        if not os.path.exists(config.LOGS_DIR):
            os.makedirs(config.LOGS_DIR)
    except OSError as e:
        raise RuntimeError(f"Failed to create logs directory: {e}")

    logger = logging.getLogger('pipeline_logger')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    try:
        fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    except Exception as e:
        raise RuntimeError(f"Failed to set up logger handlers: {e}")

    return logger