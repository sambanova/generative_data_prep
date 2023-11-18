import configparser
import logging.config
import os

from generative_data_prep.utils.utils import get_config_file_path


def test_logger_config():
    """
    GIVEN a logger configuration file path
    WHEN the logger is configured with the logger configuration file
    THEN the logger object can be properly configured
        And check if the logger object has the expected number of handlers
        And check if the logger object has the expected logging level
    """
    config_file_path = get_config_file_path()

    # Check if the file exists
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Logger configuration file not found at {config_file_path}")

    # Load the configuration
    config = configparser.ConfigParser()
    config.read(config_file_path)
    logging.config.fileConfig(config_file_path)

    # Get the logger object
    logger = logging.getLogger("generative_data_prep_logger")

    # Check if the logger object has the expected handlers, formatters, and other configurations
    assert len(logger.handlers) == 1
    assert logger.level == 10
