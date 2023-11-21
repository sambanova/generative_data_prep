import os
import secrets
import subprocess
from pathlib import Path

import pytest

PYTHON_ROOT_DIR = str(Path(__file__).parent.parent.parent.absolute())
CURRENT_DIR = str(Path(__file__).parent.absolute())


@pytest.mark.parametrize("run_path", [PYTHON_ROOT_DIR, CURRENT_DIR], ids=["python_rootdir", "home_dir"])
def test_pipeline_runs_successfully(run_path):
    """
    GIVEN a test command to run the generative_data_prep pipeline
    WHEN the command is executed in Python Root Directory
        And the command is executed in the Current Directory
    THEN the pipeline runs successfully
        And returns a return code of 0 within specified timeout
    """
    time_limit_seconds = 15
    input_file_path = PYTHON_ROOT_DIR + "/tests/examples/generative_tuning/example_generative_tuning_data.jsonl"
    output_path = f"{PYTHON_ROOT_DIR}/tests/examples/tester_{secrets.token_hex(8)}"
    os.chdir(run_path)
    command = "python -m generative_data_prep pipeline "
    command += f"--input_file_path={input_file_path} --output_path={output_path} --max_seq_length=1024"

    try:
        result = subprocess.run(command.split(), check=True, timeout=time_limit_seconds)
    except subprocess.TimeoutExpired as e:
        assert False, f"Command timed out after {time_limit_seconds} seconds. Exception: {e}"
    except Exception as e:
        assert False, f"Command failed with exception: {e}"
    assert result.returncode == 0, f"Command failed with return code {result.returncode}"
