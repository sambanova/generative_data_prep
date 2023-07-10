"""This script generates requirements files that will be executed by pre-commit as a hook and on CI."""

import argparse
import os
import re
import shlex
import subprocess  # nosec


def main():
    """The main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-version", help="python version", required=True)
    parser.add_argument("--options", help="list of dependencies to build", required=True)
    args = parser.parse_args()

    # Requirements Folders
    req_folder = "requirements"

    # Retrieve pip and pip-tools versions from the environment
    pip_op = subprocess.run(["pip", "show", "pip"], capture_output=True, check=True)  # nosec
    pip_tools_op = subprocess.run(["pip", "show", "pip-tools"], capture_output=True, check=True)  # nosec
    pip_version = re.search("Version: (.+)", pip_op.stdout.decode("utf-8")).group(1)
    pip_tools_version = re.search("Version: (.+)", pip_tools_op.stdout.decode("utf-8")).group(1)

    # Identify mount paths for docker command
    pwd = os.getcwd()

    # pip-compile common options
    pc_opts = "pip-compile --resolver=backtracking --allow-unsafe pyproject.toml --strip-extras"

    pip_compile_cmds = [f"{pc_opts} --output-file {req_folder}/requirements.txt"]
    dependency_options = args.options.split(",")
    for dep in dependency_options:
        pip_compile_cmds.append(f"{pc_opts} --extra={dep} --output-file {req_folder}/requirements-{dep}.txt")

    # Construct pip-compile commands
    pip_cmd = f"pip install pip=={pip_version} && pip install pip-tools=={pip_tools_version}"

    combo_pip_compile_cmd = f'bash -c \'{" && ".join([pip_cmd] + pip_compile_cmds)}\''
    docker_command = f"docker run \
        -e DOCKER_DEFAULT_PLATFORM=linux/amd64 \
        --rm -v {pwd}:/app -w /app \
        python:{args.python_version} {combo_pip_compile_cmd}"

    # Run pip-compile commands according to environment
    if os.getenv("CIRCLECI") == "true":
        check_diff = 'git diff --name-only HEAD^ HEAD | grep -E "^(pyproject\.toml|setup\.cfg|requirements/)"'  # noqa
        try:
            _ = subprocess.check_output(check_diff, shell=True, text=True)  # nosec
            subprocess.run(shlex.split(combo_pip_compile_cmd), check=True)  # nosec
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                print(f"Requirements Files won't be updated. No changes detected using command:\n\t{check_diff}")
            else:
                raise Exception(e)
    else:
        subprocess.run(shlex.split(docker_command), check=True)  # nosec


if __name__ == "__main__":
    main()
