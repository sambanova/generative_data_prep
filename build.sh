#!/bin/bash
#
# Build the logging library for sambanova.
#
# usage: ./build.sh
#
set -e

python3 -m venv dataprep-venv; . dataprep-venv/bin/activate
python3 -m ensurepip

python3 -m pip install --no-cache-dir build wheel
# cleanup packages
rm -rf build dist
# build wheel
python3 -m build --wheel

deactivate
