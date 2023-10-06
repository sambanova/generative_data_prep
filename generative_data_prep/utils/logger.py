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

import git

from .utils import SEP_STR

# Create a logger and set its log level
logger = logging.getLogger("generative_data_prep_logger")
# logger.setLevel(logging.INFO)  # Set your desired log level here

console_handler = logging.StreamHandler()  # This is your console handler
logger.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def log_current_datetime(logger):
    """Log the current date and time."""
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    logger.debug(f"Time Of Execution: {current_datetime}")


def log_git_commit_hash(logger):
    """Log the current git commit hash."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.debug("Running Generative Data Prep Repo: https://github.com/sambanova/generative_data_prep/")
    logger.debug(f"git commit hash: {sha}")


def log_input_args(logger, args):
    """Log the input arguments."""
    logger.debug(SEP_STR)
    logger.debug("Logging command line input flags.")
    argument_dict = vars(args)
    for arg, value in argument_dict.items():
        logger.info(f"{arg}: {value}")
