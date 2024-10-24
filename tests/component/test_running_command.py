import os
import secrets
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.conftest import PYTHON_ROOT_DIR, TESTS_EXAMPLES_PATH

CURRENT_DIR = str(Path(__file__).parent.absolute())


@contextmanager
def change_directory_for_testing(new_path):
    """Change directory for testing.

    Context manager that changes the directory to the new path and then changes it back to the original path.

    Args:
        new_path (str): The new path to change to.
    """
    original_cwd = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@pytest.mark.slow
@pytest.mark.parametrize("run_path", [PYTHON_ROOT_DIR, CURRENT_DIR], ids=["python_rootdir", "home_dir"])
def test_pipeline_runs_successfully(run_path):
    """
    GIVEN a test command to run the generative_data_prep pipeline
    WHEN the command is executed in Python Root Directory
        And the command is executed in the Current Directory
    THEN the pipeline runs successfully
        And returns a return code of 0 within specified timeout
    """
    # TODO: Need to reduce the time limit back down to 9 or get it close to 9
    time_limit_seconds = 19
    input_path = TESTS_EXAMPLES_PATH / "generative_tuning" / "example_generative_tuning_data.jsonl"
    output_path = TESTS_EXAMPLES_PATH / f"tester_{secrets.token_hex(8)}"
    with change_directory_for_testing(run_path):
        command = "python -m generative_data_prep pipeline "
        command += f"--input_path={input_path} --output_path={output_path} "
        command += "--max_seq_length=1024 --pretrained_tokenizer=openai-community/gpt2"

        try:
            result = subprocess.run(command.split(), check=True, timeout=time_limit_seconds)
        except subprocess.TimeoutExpired as e:
            assert False, f"Command timed out after {time_limit_seconds} seconds. Exception: {e}"
        except Exception as e:
            assert False, f"Command failed with exception: {e}"
        assert result.returncode == 0, f"Command failed with return code {result.returncode}"
