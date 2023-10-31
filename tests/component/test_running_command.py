import secrets
import subprocess


def test_pipeline_runs_successfully(pytestconfig):
    """
    GIVEN a test command to run the generative_data_prep pipeline
    WHEN the command is executed using python
    THEN the pipeline runs successfully and returns a return code of 0 within specified timeout
    """
    time_limit_seconds = 9
    root_dir = pytestconfig.rootdir
    input_file_path = root_dir + "/tests/examples/generative_tuning/example_generative_tuning_data.jsonl"
    output_path = f"{root_dir}/tests/examples/tester_{secrets.token_hex(8)}"
    command = "python -m generative_data_prep pipeline "
    command += f"--input_file_path={input_file_path} --output_path={output_path} --max_seq_length=1024"

    try:
        result = subprocess.run(command.split(), check=True, timeout=time_limit_seconds)
    except subprocess.TimeoutExpired as e:
        assert False, f"Command timed out after {time_limit_seconds} seconds. Exception: {e}"
    except Exception as e:
        assert False, f"Command failed with exception: {e}"
    assert result.returncode == 0, f"Command failed with return code {result.returncode}"
