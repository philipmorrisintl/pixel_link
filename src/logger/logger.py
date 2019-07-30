"""
    Original module author: Patrick Kennedy <patkennedy79@gmail.com>
"""


from typing import Optional
import logging
import logging.config as logging_config
import os
import warnings

from src.logger.utils import adjust_config


warnings.simplefilter(action='ignore', category=FutureWarning)
config = adjust_config()
logging_config.fileConfig(config)


def get_logger(logger_file_name: Optional[str] = None) -> logging.Logger:
    """
    Used to get logger object

    :param logger_file_name: Log filename
    :return: Logger object
    """
    if logger_file_name is not None:
        logger = logging.getLogger(os.path.basename(logger_file_name))
    else:
        logger = logging.getLogger('UnnamedLogger')
    return logger


