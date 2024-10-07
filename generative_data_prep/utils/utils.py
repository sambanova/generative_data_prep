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
"""

import hashlib
import json
import math
import os
import shutil
from subprocess import PIPE, run  # nosec
from typing import Optional


def execute_and_return_stdout(command):
    """Execute [command] using os.system, and then returns the terminal outputs.

    The text can be accessed by accessing [result].stderr or [result].stdout

    Args:
        command (str): string format of linux command to execute

    Returns:
        Piped Out object: Access text using .stout or .stderr attributes of output object
    """
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)  # nosec
    return result


def _calculate_sha256(file_path: str):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_walk_files_to_hash(dir: str, filter: Optional[str] = None):
    files_to_hash = []
    for foldername, _, filenames in os.walk(dir):
        if filter is not None and filter in foldername.split("/"):
            continue
        relative_foldername = os.path.relpath(foldername, dir)
        if relative_foldername == ".":
            relative_foldername = ""
        relative_foldername = relative_foldername.replace(os.path.sep, "_")
        if relative_foldername != "":
            relative_foldername += "_"
        hash_file_names = [
            (
                os.path.join(foldername, filename),
                relative_foldername + filename,
            )
            for filename in filenames
        ]

        files_to_hash.extend(hash_file_names)
    return files_to_hash


def validate_sha256(output_dir: str):
    """Validates the current sha256 directory within the output directory.

    Args:
        output_dir (str)

    Returns:
        Boolean value: False for the files have been modified, and True for all
        files have not been modified
    """
    files_to_hash = _get_walk_files_to_hash(output_dir, "sha256")
    sha_info_file = os.path.join(output_dir, "sha256", "files_metadata.json")
    with open(sha_info_file, "r") as output_file:
        file_info_dict = json.load(output_file)
    for file, hash_file_name in files_to_hash:
        if hash_file_name not in file_info_dict:
            print("Error: The hash for this file was not found in the metadata.")
            print(f"Missing hash file: {hash_file_name}")
            print(f"Available metadata: {list(file_info_dict.keys())}")
            return False
        current_modified_time = os.path.getmtime(file)
        current_size = os.path.getsize(file)
        if current_size != file_info_dict[hash_file_name]["size"] or not math.isclose(
            current_modified_time, file_info_dict[hash_file_name]["modified_time"]
        ):
            file_hash = file_info_dict[hash_file_name]["sha256"]
            current_file_hash = _calculate_sha256(file)
            if file_hash != current_file_hash:
                print("Error: File has been modified or the SHA256 hash does not match.")
                print(f"Expected hash: {file_hash}")
                print(f"Actual hash: {current_file_hash}")
                return False
    return True


def create_sha256(output_dir: str):
    """Creates the corresponding sha256 files for each of the files within the output directory.

    Args:
        output_dir (str)

    Returns:
        None
    """
    hash_dir = os.path.join(output_dir, "sha256")
    if os.path.isdir(hash_dir):
        shutil.rmtree(hash_dir)
    files_to_hash = _get_walk_files_to_hash(output_dir)
    os.mkdir(hash_dir)
    output_file_hash = os.path.join(hash_dir, "files_metadata.json")
    file_info_dict = {}
    for file, hash_file_name in files_to_hash:
        file_hash = _calculate_sha256(file)
        file_info_dict[hash_file_name] = {
            "sha256": file_hash,
            "size": os.path.getsize(file),
            "modified_time": os.path.getmtime(file),
        }
    with open(output_file_hash, "w") as output_file:
        json.dump(file_info_dict, output_file)


def get_config_file_path():
    """Return absolute path to the logging config file.

    Returns:
        Path: absolute path to the logging config file
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_filename = "configs/logger.conf"  # Change this to match your config file name
    config_path = os.path.join(script_dir, config_filename)
    return config_path


def get_num_training_splits(input_file_size_in_gb: float, num_training_splits: Optional[int] = None) -> int:
    """Determines the number of training splits based on the size of the input file in gigabytes.

    Parameters:
    input_file_size_in_gb : float
        The size of the input file in gigabytes.
    num_training_splits : int, optional
        The number of training splits to use. If not provided, a default value is determined
        based on the input file size.

    Returns:
    int
        The number of training splits.
    """
    # Determine number of training splits if not provided
    if num_training_splits is None:
        # Default number of splits based on file size
        if input_file_size_in_gb < 10:
            num_training_splits = 32
        elif input_file_size_in_gb < 100:
            num_training_splits = 128
        else:
            num_training_splits = 256

    return num_training_splits
