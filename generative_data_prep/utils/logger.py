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
import os

import git

try:
    SEP_STR = "-" * os.get_terminal_size().columns
except OSError:
    SEP_STR = "----------------------------------------------------------------------------------"


class Logger(object):
    """Singleton class to ensure that only one instance of a logger is created."""

    def __new__(cls):
        """Create a new Logger object if it does not exist."""
        if not hasattr(cls, "instance"):
            breakpoint()
            cls.instance = super(Logger, cls).__new__(cls)
            cls.instance._logger = logging.getLogger("generative_data_prep_logger")
            cls.instance._logger.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()  # This is your console handler
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(formatter)
            cls.instance._logger.addHandler(console_handler)
            return cls.instance
        return cls.instance

    @classmethod
    def info(cls, message, print_sep_str=True):
        """Log at the info level."""
        if print_sep_str:
            cls._logger(SEP_STR)
        cls._logger.info(message)

    @classmethod
    def error(cls, message, print_sep_str=True):
        """Log at the error level."""
        if print_sep_str:
            cls._logger(SEP_STR)
        cls._logger.info(message)

    @classmethod
    def warning(cls, message, print_sep_str=True):
        """Log at the warning level."""
        if print_sep_str:
            cls._logger(SEP_STR)
        cls._logger.info(message)

    @classmethod
    def debug(cls, message, print_sep_str=True):
        """Log at the debug level."""
        if print_sep_str:
            cls._logger(SEP_STR)
        cls._logger.info(message)

    @classmethod
    def log_current_datetime(cls):
        """Log at the current date and time."""
        current_datetime = datetime.datetime.now()
        cls._logger.debug(f"Time of execution: {current_datetime}")

    @classmethod
    def log_git_commit_hash(cls):
        """Log the current git commit hash."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        cls._logger.debug("Running Generative Data Prep repository: https://github.com/sambanova/generative_data_prep/")
        cls._logger.debug(f"Git commit hash: {sha}")

    @classmethod
    def log_input_args(cls, args):
        """Log the input arguments."""
        cls._logger.debug(SEP_STR)
        cls._logger.debug("Logging command line input flags.")
        argument_dict = vars(args)
        for arg, value in argument_dict.items():
            cls._logger.debug(f"{arg}: {value}")

    @classmethod
    def add_handler(cls, handler):
        """Add a handler to the logger."""
        cls._logger.addHandler(handler)

    @classmethod
    def log_metrics(cls, metrics):
        """Log the metrics table."""
        cls._logger.info(f"{cls._header('Metrics')}\n{metrics}\n{cls._header('Complete')}")

    @classmethod
    def _header(cls, header_name: str):
        """Create a header out of the header_name string."""
        half_sep_str = int((len(SEP_STR) - len(header_name)) / 2) * "-"
        return half_sep_str + header_name + half_sep_str


logger = Logger()
