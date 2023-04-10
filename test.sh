#!/usr/bin/env bash

set -e

# docstring checks
# TODO: add generative_data_prep/processing_functions
pydocstyle generative_data_prep/__main__.py generative_data_prep/utils/path_verify.py generative_data_prep/data_prep/data_prep.py generative_data_prep/data_prep/pipeline.py generative_data_prep/utils/balance_hdf5_files.py generative_data_prep/utils/arg_configs.py
python -m pytest --doctest-modules generative_data_prep/__main__.py generative_data_prep/tokenized_line generative_data_prep/processors generative_data_prep/data_buffers generative_data_prep/utils/arg_configs.py generative_data_prep/utils/path_verify.py generative_data_prep/data_prep/data_prep.py generative_data_prep/data_prep/pipeline.py generative_data_prep/utils/balance_hdf5_files.py
# # style checks
flake8 generative_data_prep/__main__.py generative_data_prep/processors generative_data_prep/tokenized_line generative_data_prep/data_buffers tests generative_data_prep/utils/arg_configs.py generative_data_prep/data_prep/data_prep.py generative_data_prep/data_prep/pipeline.py
# type checks
mypy generative_data_prep/__main__.py generative_data_prep/processors generative_data_prep/tokenized_line generative_data_prep/data_buffers tests generative_data_prep/utils/path_verify.py generative_data_prep/utils/arg_configs.py
# tests
python -m coverage run --source=generative_data_prep/ -m pytest tests
