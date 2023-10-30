from generative_data_prep.utils.utils import get_config_file_path


def test_get_config_file_path():
    """
    GIVEN a function named `get_config_file_path`
    WHEN the function is called
    THEN it should return the correct config file path
    """

    assert str(get_config_file_path()).split("/")[-1] == "logger.conf"
