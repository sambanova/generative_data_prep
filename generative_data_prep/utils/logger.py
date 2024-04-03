"""Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This class creates a common logger.
"""
import datetime
import logging
import logging.config
import os
from typing import Dict, Union

import git

LOGGER = logging.getLogger("generative_data_prep_logger")
START_TIME = datetime.datetime.now()

try:
    SEP_STR = "-" * os.get_terminal_size().columns
except OSError:
    SEP_STR = "-" * 80


def add_file_handler(log_file_path: str, output_dir: str):
    """If log_file_path is defined then return it, otherwise return output_dir/logs.log.

    Args:
        log_file_path: The input log_file_path flag.
        output_dir: The output directory to default to if log_file_path is None.
    """
    if log_file_path is None:
        log_file_path = os.path.join(output_dir, "logs.log")
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_file_path, "w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    # Add the file handler to the Logger
    LOGGER.addHandler(file_handler)


def log_current_datetime():
    """Log at the current date and time."""
    current_datetime = datetime.datetime.now()
    LOGGER.debug(SEP_STR)
    LOGGER.debug(f"Time of execution: {current_datetime}")


def log_git_commit_hash():
    """Log the current git commit hash."""
    LOGGER.debug("Running Generative Data Prep repository: https://github.com/sambanova/generative_data_prep/")
    try:
        repo = git.Repo(os.path.abspath(__file__), search_parent_directories=True)
        sha = repo.head.object.hexsha
        LOGGER.debug(f"Git commit hash: {sha}")
    except git.exc.InvalidGitRepositoryError:
        LOGGER.debug("Git commit hash not found")


def log_input_args(args):
    """Log the input arguments."""
    LOGGER.debug(SEP_STR)
    LOGGER.debug("Logging command line input flags.")
    argument_dict = vars(args)
    for arg, value in argument_dict.items():
        LOGGER.debug(f"{arg}: {value}")


def log_metrics(metrics):
    """Log the metrics table."""
    LOGGER.info(f"{get_header('Metrics')}\n{metrics}\n{get_header('Complete')}")


def get_header(header_name: str):
    """Create a header out of the header_name string."""
    half_sep_str = int((len(SEP_STR) - len(header_name)) / 2) * "-"
    return half_sep_str + header_name + half_sep_str


def log_elapsed_time():
    """Log how much time it took to execute entire script."""
    LOGGER.info(f"Elapsed time: {datetime.datetime.now().replace(microsecond=0) - START_TIME.replace(microsecond=0)}")


def log_training_details(dataset_metadata: Dict[str, Union[str, int, bool]]):
    """Log training parameters that need to be used with this dataset."""
    LOGGER.info(SEP_STR)
    LOGGER.info("The dataset you have prepared requires you to set the following training parameters:")
    LOGGER.info(f"    Max sequence length == {dataset_metadata['max_seq_length']}")
    LOGGER.info(f"    Model vocabulary size == {dataset_metadata['vocab_size']}")
    LOGGER.info(f"    Local batch size <= {dataset_metadata['max_batch_size_train']}")
    LOGGER.info(f"    Number of RDUs (data parallel workers) {dataset_metadata['number_of_training_files']}")
    LOGGER.info(f"    Do eval: {'True' if int(dataset_metadata['number_of_dev_files']) >= 1 else 'False'}")


def log_sep_str():
    """Log the seperator string."""
    LOGGER.info(SEP_STR)
