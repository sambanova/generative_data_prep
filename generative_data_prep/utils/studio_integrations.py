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


This module provides utility functions for data preparation in a machine learning pipeline.

Key components:
- get_max_seq_length_arg: Extracts max sequence length from model config.
- get_shuffle_arg: Determines shuffling method based on file size.
- adjust_splits: Ensures the number of training splits is compatible with RDUs.
- verify_enough_data_to_run_one_batch: Checks if there's sufficient data for training.
- training_to_data_prep_params: Converts training params to data prep params.
"""
import os
from typing import Optional

from transformers import AutoConfig, AutoTokenizer

from .arg_parser import get_arg_parser
from .utils import get_num_training_splits


def get_max_seq_length_arg(model_config):
    """Retrieves the maximum sequence length from a model configuration object.

    Args:
        model_config: The configuration object of the model, typically loaded from a pretrained model checkpoint.

    Returns:
        int or str: The maximum sequence length of the model. This can be extracted from either the
        `max_position_embeddings` or `n_positions` attribute of the `model_config`.
        If neither attribute is found, a message indicating "Sequence length not found in config" is returned.
        In case of an exception, the error message is returned as a string.

    Raises:
        None: This function handles exceptions and returns an error message in case of failure.
    """
    try:
        # Extract the max sequence length
        if hasattr(model_config, "max_position_embeddings"):
            return model_config.max_position_embeddings
        elif hasattr(model_config, "n_positions"):
            return model_config.n_positions
        else:
            return "Sequence length not found in config"
    except Exception as e:
        return f"Error: {str(e)}"


def get_shuffle_arg(input_file_path: str):
    """Checks if the file size is less than 2GB.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: 'on_RAM' if file is less than 2GB, 'large_file' otherwise.
    """
    # Get the file size in bytes
    file_size = os.path.getsize(input_file_path)

    # Convert 2GB to bytes (2 * 1024^3)
    size_limit = 2 * 1024 * 1024 * 1024

    if file_size < size_limit:
        return "on_RAM"
    else:
        return "large_file"


def adjust_splits(num_training_splits: int, number_of_rdus: int):  # noqa: D417
    """Adjusts the number of training splits to be a positive multiple of the number of RDUs.

    This function verifies that `num_training_splits` is a positive multiple of `number_of_rdus`.
    If it is not, the function adjusts `num_training_splits` to the smallest positive multiple of `number_of_rdus`.

    Parameters:
        num_training_splits (int): The number of training splits to be adjusted.
        number_of_rdus (int): The number of RDUs to be used as the divisor.

    Returns:
        int: The adjusted number of training splits, which is a positive multiple of `number_of_rdus`.
    """  # noqa: D417
    if num_training_splits <= 0 or number_of_rdus <= 0:
        raise ValueError("Both num_training_splits and number_of_rdus must be positive integers.")

    # Check if num_training_splits is already a multiple of number_of_rdus
    if num_training_splits % number_of_rdus == 0:
        return num_training_splits
    else:
        # Find the smallest multiple of number_of_rdus greater than num_training_splits
        multiple = (num_training_splits // number_of_rdus + 1) * number_of_rdus
        return multiple


def verify_enough_data_to_run_one_batch(
    input_path: str,
    num_training_splits: int,
    grad_accum_steps: int,
    pef_batch_size: int,
    max_seq_length: int,
    number_of_rdus: int,
) -> int:
    """Verifies if there is enough data in the input file to run one batch of training.

    This function checks if the provided file contains enough bytes to process at least
    one batch of training data based on the specified parameters. If not enough data
    is found, it recursively reduces the number of training splits until an acceptable
    configuration is found or raises an error if it's not possible.

    Args:
        input_path (str): Path to the input data file.
        num_training_splits (int): Number of training splits to use.
        grad_accum_steps (int): Number of gradient accumulation steps.
        pef_batch_size (int): Batch size used for the training.
        max_seq_length (int): Maximum sequence length per input.
        number_of_rdus (int): Number of RDUs.

    Returns:
        int: The number of training splits if there is enough data.

    Raises:
        ValueError: If there is not enough data to run one batch,
                    and reducing training splits does not resolve the issue.
    """
    # Calculate the minimum number of bytes required for one batch
    min_required_bytes = (
        num_training_splits * grad_accum_steps * pef_batch_size * max_seq_length * 3
    )  # 3 bytes per token

    try:
        # Open the file and check if the file has enough data
        with open(input_path, "rb") as f:
            # Seek to the byte position just before min_required_bytes
            f.seek(min_required_bytes - 1)
            byte = f.read(1)  # Read the next byte to check if data exists

            if byte:  # If a byte is returned, enough data is available
                return num_training_splits
            else:
                # If no byte is found and training splits can be reduced
                if num_training_splits > number_of_rdus * 2:
                    num_training_splits = number_of_rdus * 2
                    return verify_enough_data_to_run_one_batch(
                        input_path,
                        num_training_splits,
                        grad_accum_steps,
                        pef_batch_size,
                        max_seq_length,
                        number_of_rdus,
                    )
                else:
                    # If it's not possible to reduce further, raise an error
                    err_msg = (
                        f"{input_path} does not have enough data to run one batch of training.\n"
                        "To resolve this issue, you may:\n"
                        "  - Increase the amount of text data in input_path.\n"
                        "  - Decrease the number of RDUs.\n"
                        "  - Decrease the gradient accumulation steps."
                    )
                    raise ValueError(err_msg)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {input_path} does not exist.")


def training_to_data_prep_params(  # noqa: C901
    input_path: str,
    output_path: str,
    log_file_path: str,
    checkpoint_path: str,
    number_of_rdus: int,
    grad_accum_steps: int,
    pef_batch_size: int,
    max_seq_length: Optional[int] = None,
    evaluation_ratio: Optional[float] = None,
    num_workers: int = 16,
    custom_tokenizer_path: Optional[str] = None,
    input_packing_config: str = "greedy::drop",
    apply_chat_template: Optional[bool] = None,
    shuffle: Optional[str] = None,
):  # noqa: C901
    """Convert training hyper-parameters to data prep hyper-parameters.

    Args:
        input_path (str): Path to the input data file.
        output_path (str): Path to save the processed output.
        log_file_path (str): Path to the log file for storing training logs.
        checkpoint_path (str): Path to the model checkpoint for loading the tokenizer and config.
        number_of_rdus (int): Number of RDUs to be used for parallel processing.
        grad_accum_steps (int): Number of gradient accumulation steps.
        pef_batch_size (int): Size of each batch for PEF.
        evaluation_ratio (float, optional): Ratio for splitting data into evaluation sets.
        num_workers (int, optional): Number of workers for data loading. Defaults to 16.
        custom_tokenizer_path (str, optional): Path to a custom tokenizer, if any. If not provided,
            defaults to the tokenizer from `checkpoint_path`.
        input_packing_config (str, optional): Strategy for packing input sequences.
            Defaults to "greedy::drop".
        apply_chat_template (bool, optional): Whether to apply chat formatting to inputs.
            If None, it is inferred based on the tokenizer's capabilities. Defaults to None.

    Returns:
        Namespace: Parsed arguments for the data preparation pipeline.

    Raises:
        ValueError: If the tokenizer's vocabulary size exceeds the model's vocabulary size,
            or if a custom tokenizer class does not match the base checkpoint tokenizer class.
    """
    if custom_tokenizer_path is not None:
        pretrained_tokenizer = custom_tokenizer_path
    else:
        pretrained_tokenizer = checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    model_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Validate that the tokenizer vocab size is <= model vocab size
    if not tokenizer.vocab_size <= model_config.vocab_size:
        err_msg = f"Tokenizers vocab size: {tokenizer.vocab_size}"
        err_msg += " is not compatible with model vocab size: {model_config.vocab_size}"
        raise ValueError(err_msg)
    if custom_tokenizer_path is not None and not isinstance(tokenizer, AutoTokenizer.from_pretrained(checkpoint_path)):
        custom_tok_type = type(tokenizer)
        base_ckpt_tok_type = type(AutoTokenizer.from_pretrained(checkpoint_path))
        raise ValueError(
            f"The custome tokenizer class({custom_tok_type}) is not the same as base checkpoint({base_ckpt_tok_type})"
        )

    if apply_chat_template is None:
        if not hasattr(tokenizer, "apply_chat_template"):
            apply_chat_template = False
        else:
            try:
                test = tokenizer.apply_chat_template([{"role": "user", "content": "Hi."}])
                apply_chat_template = True
            except ValueError:
                apply_chat_template = False

    input_file_size_in_bytes = os.stat(input_path).st_size
    input_file_size_in_gb = input_file_size_in_bytes / (1024**3)

    num_training_splits = get_num_training_splits(input_file_size_in_gb)
    # verify that num_training_splits is a positive multiple of the number_of_rdus
    # if not a positive multiple then find the smallest positive multiple larger then the default
    num_training_splits = adjust_splits(num_training_splits, number_of_rdus)

    if shuffle is None:
        shuffle = get_shuffle_arg(input_path)

    if max_seq_length is None:
        max_seq_length = get_max_seq_length_arg(model_config)

    # verify that there are approximately enough lines to have enough data to
    # run with this configuration where the amount of data needed to run one batch is
    # Min number of bytes in file =
    # num_training_splits * grad_accum_steps * pef_batch_size * max_seq_length * 3 (avg bytes per token)
    num_training_splits = verify_enough_data_to_run_one_batch(
        input_path, num_training_splits, grad_accum_steps, pef_batch_size, max_seq_length, number_of_rdus
    )

    if evaluation_ratio is not None:
        num_dev_splits = None
        num_test_splits = None
        dev_ratio = evaluation_ratio
        test_ratio = 0.0
    else:
        num_dev_splits = 0
        num_test_splits = 0
        dev_ratio = None
        test_ratio = None

    arg_parser = get_arg_parser()

    input_arguments = ["pipeline"]
    actions = arg_parser._subparsers._actions[1].choices["pipeline"]._actions  # type: ignore
    arg_names = [action.dest for action in actions]
    for local_var_name, local_var_val in locals().items():
        if local_var_name in arg_names and local_var_val is not None:
            if not isinstance(local_var_val, bool):
                input_arguments.append(f"--{local_var_name}")
                input_arguments.append(str(local_var_val))
            elif local_var_val:
                input_arguments.append(f"--{local_var_name}")

    data_prep_args = arg_parser.parse_args(input_arguments)

    return data_prep_args
