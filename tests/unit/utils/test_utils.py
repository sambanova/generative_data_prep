import os
from generative_data_prep.utils.utils import get_config_file_path


def test_get_config_file_path():
    """
    GIVEN a function named `get_config_file_path`
    WHEN the function is called
    THEN it should return the correct config file path
    """
    full_config_file_path = get_config_file_path()
    expected_config_filename = "logger.conf"
    assert (
        str(full_config_file_path).split("/")[-1] == expected_config_filename
    ), f"The returned file path is not {expected_config_filename}."
    assert os.path.exists(full_config_file_path), f"The returned file path {full_config_file_path} does not exist."
