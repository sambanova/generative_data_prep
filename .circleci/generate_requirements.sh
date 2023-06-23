#!/bin/bash

set -eo pipefail

# Identify mount paths for docker command
PWD=$(pwd)
PIP_CACHE_DIR=$(pip cache dir)

# pip-compile options
PC_OPTS="--resolver=backtracking --allow-unsafe pyproject.toml --output-file=requirements"

# Construct pip-compile commands
PIP_COMMAND="pip install -U pip && pip install -U pip-tools==${PIP_TOOLS_VER}"
PIP_COMPILE_COMMAND="\
    pip-compile ${PC_OPTS}/requirements.txt \
    && pip-compile --extra=build ${PC_OPTS}/build-requirements.txt \
    && pip-compile --extra=dev ${PC_OPTS}/dev-requirements.txt \
    && pip-compile --extra=docs ${PC_OPTS}/docs-requirements.txt \
    && pip-compile --extra=tests ${PC_OPTS}/tests-requirements.txt"
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
