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


Helper functions to verify that file paths are valid.

Check to make sure that the files can be created and correctly written or read from.
If the path is not valid, then fail out gracefully with ValueError.
"""
import logging
import os
import shutil
from typing import Optional

from .utils import SEP_STR


def verify_input_file(input_file: str):
    """Verify that the input file is valid path and is readable.

    Args:
        input_file: path to input file

    Raises:
        ValueError: If the input file is not valid
    """
    if not os.path.isfile(input_file):
        raise ValueError(f"{input_file} is not a valid input path")

    if not os.access(input_file, os.R_OK):
        raise ValueError(f"{input_file} is not readable")


def verify_output_file(output_file: str, overwrite_output_dir: bool):
    """Verify that the output file path is a valid and is writeable.

    Args:
        output_file: path to output file
        overwrite_output_dir: If the output file can be over-ridden

    Raises:
        ValueError: If the output file is not valid
    """
    if os.path.exists(output_file):
        if not overwrite_output_dir:
            err_msg = f"Your output path {output_file} already exists - "
            err_msg += "if you want to over-write this file please use the argument --overwrite_output_path"
            raise ValueError(err_msg)

    if not os.path.exists(os.path.dirname(output_file)):
        raise ValueError(f"The output path {output_file} is not valid - {os.path.dirname(output_file)} does not exist")

    if not os.access(os.path.dirname(output_file), os.W_OK):
        raise ValueError(f"The output path {output_file} is not writeable")


def verify_output_dir(
    output_dir: str,
    overwrite_output_dir: bool,
    raise_warning_if_exists: Optional[bool] = True,
):
    """Verify that the output directory is valid and writable.

    Args:
        output_dir: path to output directory
        overwrite_output_dir: whether to delete everything in output directory
        raise_warning_if_exists: Print warning if path exists and do overwrite_output_dir is False.
        Defaults to True.

    Raises:
        ValueError: If the output directory is not valid
    """
    if os.path.exists(output_dir):
        if overwrite_output_dir:
            for file_name in os.listdir(output_dir):
                if "cache" not in file_name:
                    file_path = os.path.join(output_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    else:
                        shutil.rmtree(file_path)
        elif raise_warning_if_exists:
            logging.info(SEP_STR)
            logging.warning(f"WARNING: {output_dir} already exists, new files will be written here")
    else:
        os.makedirs(output_dir)

    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"The cache directory you provided {output_dir} is not writeable")
