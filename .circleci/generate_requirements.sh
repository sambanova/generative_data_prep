#!/bin/bash

set -eo pipefail

# Identify mount paths for docker command
PWD=$(pwd)
PIP_CACHE_DIR=$(pip cache dir)

# Construct pip-compile commands
PIP_COMMAND="pip install -U pip && pip install -U pip-tools==${PIP_TOOLS_VER}"
PIP_COMPILE_COMMAND="\
    pip-compile --output-file=requirements/requirements.txt pyproject.toml \
    && pip-compile --resolver=backtracking --extra=build --output-file=requirements/requirements_build.txt pyproject.toml \
    && pip-compile --resolver=backtracking --extra=dev --output-file=requirements/requirements_dev.txt pyproject.toml \
    && pip-compile --resolver=backtracking --extra=docs --output-file=requirements/requirements_docs.txt pyproject.toml \
    && pip-compile --resolver=backtracking --extra=tests --output-file=requirements/requirements_tests.txt pyproject.toml"
PLAIN_COMMAND="${PIP_COMMAND} && ${PIP_COMPILE_COMMAND}"
DOCKER_COMMAND="docker run --rm -v ${PWD}:/app -v ${PIP_CACHE_DIR}:/root/.cache/pip -w /app python:${PYTHON_VER} bash -c '${PLAIN_COMMAND}'"

# Run pip-compile commands according to environment
if [ "$CIRCLECI" = "true" ]; then
    echo "Generating Requirements in Python venv for CircleCI"
    bash -c "${PLAIN_COMMAND}"
else
    echo "Generating Requirements in Docker for Local Environment"
    bash -c "${DOCKER_COMMAND}"
fi
