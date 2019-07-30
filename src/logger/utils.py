import os
from configparser import ConfigParser

CONFIG_FILE_PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'logging.ini'
)
LOGGING_FILE_PLACEHOLDER = '{logging_file_path}'
TARGET_LOG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..',
    '..',
    'logs',
    'logs.log'
)


def adjust_config() -> ConfigParser:
    config = _load_config()
    return _set_up_logger_target_file(config)


def _load_config() -> ConfigParser:
    config = ConfigParser()
    config.read(CONFIG_FILE_PATH)
    return config


def _set_up_logger_target_file(config: ConfigParser) -> ConfigParser:
    file_handler_args = config['handler_fileHandler']['args']
    file_handler_args = file_handler_args.replace(
        LOGGING_FILE_PLACEHOLDER,
        TARGET_LOG_PATH
    )
    config['handler_fileHandler']['args'] = file_handler_args
    return config
