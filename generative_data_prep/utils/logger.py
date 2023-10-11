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
from pathlib import Path

import git
import yaml

try:
    SEP_STR = "-" * os.get_terminal_size().columns
except OSError:
    SEP_STR = "-" * 80


class Logger(object):
    """Singleton class to ensure that only one instance of a logger is created."""

    def __new__(cls):
        """Create a new Logger object if it does not exist."""
        if not hasattr(cls, "instance"):
            cls.instance = super(Logger, cls).__new__(cls)
            with open(os.path.join(Path.cwd(), "generative_data_prep/utils/logger_config.yaml"), "rt") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            cls.instance = logging.getLogger("generative_data_prep_logger")
        return cls.instance

    @classmethod
    def add_file_handler(cls, log_file_path: str, output_dir: str):
        """If log_file_path is defined then return it, otherwise return output_dir/logs.log.

        Args:
            log_file_path: The input log_file_path flag.
            output_dir: The output directory to default to if log_file_path is None.
        """
        if log_file_path is None:
            log_file_path = os.path.join(output_dir, "logs.log")
        formatter = logging.Formatter("%(message)s")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        # Add the file handler to the Logger()
        cls.instance.addHandler(file_handler)

    @classmethod
    def info(cls, message, print_sep_str=True):
        """Log at the info level."""
        if print_sep_str:
            cls.instance(SEP_STR)
        cls.instance.info(message)

    @classmethod
    def error(cls, message, print_sep_str=True):
        """Log at the error level."""
        if print_sep_str:
            cls.instance.info(SEP_STR)
        cls.instance.info(message)

    @classmethod
    def warning(cls, message, print_sep_str=True):
        """Log at the warning level."""
        if print_sep_str:
            cls.instance.info(SEP_STR)
        cls.instance.info(message)

    @classmethod
    def debug(cls, message, print_sep_str=True):
        """Log at the debug level."""
        if print_sep_str:
            cls.instance.info(SEP_STR)
        cls.instance.info(message)

    @classmethod
    def log_current_datetime(cls):
        """Log at the current date and time."""
        current_datetime = datetime.datetime.now()
        cls.instance.debug(f"Time of execution: {current_datetime}")

    @classmethod
    def log_git_commit_hash(cls):
        """Log the current git commit hash."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        cls.instance.debug(
            "Running Generative Data Prep repository: https://github.com/sambanova/generative_data_prep/"
        )
        cls.instance.debug(f"Git commit hash: {sha}")

    @classmethod
    def log_input_args(cls, args):
        """Log the input arguments."""
        cls.instance.debug(SEP_STR)
        cls.instance.debug("Logging command line input flags.")
        argument_dict = vars(args)
        for arg, value in argument_dict.items():
            cls.instance.debug(f"{arg}: {value}")

    @classmethod
    def log_metrics(cls, metrics):
        """Log the metrics table."""
        cls.instance.info(f"{cls._header('Metrics')}\n{metrics}\n{cls._header('Complete')}")

    @classmethod
    def _header(cls, header_name: str):
        """Create a header out of the header_name string."""
        half_sep_str = int((len(SEP_STR) - len(header_name)) / 2) * "-"
        return half_sep_str + header_name + half_sep_str
