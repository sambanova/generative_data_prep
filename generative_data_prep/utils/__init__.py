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

from .add_metadata_to_dataset import (
    METADATA_KEYS_CANT_ADD,
    add_all_metadata_to_dataset,
    add_all_metadata_to_dir_of_datasets,
    add_seq_metadata_dataset,
    add_seq_metadata_to_dir_of_datasets,
)
from .arg_configs import PackingConfig
from .arg_parser import get_arg_parser
from .balance_hdf5_files import balance_hdf5_files
from .constants import (
    CATEGORY_JSON_KEY,
    BoundaryType,
    FileExtension,
    OverflowType,
    PackingStyleType,
    TokenTypeIds,
)
from .large_file_shuffle import large_file_shuffle
from .logger import (
    add_file_handler,
    check_deprecated_args,
    log_current_datetime,
    log_elapsed_time,
    log_git_commit_hash,
    log_input_args,
    log_installed_packages,
    log_metrics,
    log_sep_str,
    log_training_details,
)
from .metadata_generation import DatasetMetadata
from .path_verify import verify_input_file, verify_output_dir, verify_output_file
from .studio_integrations import (
    adjust_splits,
    get_max_seq_length_arg,
    get_shuffle_arg,
    training_to_data_prep_params,
    verify_enough_data_to_run_one_batch,
)
from .utils import (
    create_sha256,
    execute_and_return_stdout,
    get_config_file_path,
    get_num_training_splits,
    save_tokenizer,
    validate_sha256,
)

__all__ = [
    "PackingConfig",
    "balance_hdf5_files",
    "BoundaryType",
    "FileExtension",
    "OverflowType",
    "PackingStyleType",
    "TokenTypeIds",
    "large_file_shuffle",
    "verify_input_file",
    "verify_output_dir",
    "verify_output_file",
    "execute_and_return_stdout",
    "create_sha256",
    "validate_sha256",
    "CATEGORY_JSON_KEY",
    "add_file_handler",
    "log_input_args",
    "log_current_datetime",
    "log_metrics",
    "log_git_commit_hash",
    "log_elapsed_time",
    "log_training_details",
    "log_sep_str",
    "log_installed_packages",
    "get_config_file_path",
    "DatasetMetadata",
    "check_deprecated_args",
    "training_to_data_prep_params",
    "get_arg_parser",
    "get_num_training_splits",
    "get_max_seq_length_arg",
    "get_shuffle_arg",
    "adjust_splits",
    "verify_enough_data_to_run_one_batch",
    "save_tokenizer",
    "add_seq_metadata_dataset",
    "add_seq_metadata_to_dir_of_datasets",
    "add_all_metadata_to_dataset",
    "add_all_metadata_to_dir_of_datasets",
    "METADATA_KEYS_CANT_ADD",
]
