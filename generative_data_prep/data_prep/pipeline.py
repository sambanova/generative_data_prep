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


Data preparation pipeline for converting a jsonl file to tokenized hdf5 files consumable by SambaSuite.
"""

import concurrent.futures
import json
import logging
import multiprocessing
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from sys import platform
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import psutil
import yaml
from alive_progress import alive_bar
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main
from generative_data_prep.processors.metrics import Metrics
from generative_data_prep.utils import (
    BoundaryType,
    PackingConfig,
    balance_hdf5_files,
    create_sha256,
    execute_and_return_stdout,
    get_num_training_splits,
    large_file_shuffle,
    log_sep_str,
    save_tokenizer,
    verify_input_file,
    verify_output_dir,
    verify_output_file,
)

LOGGER = logging.getLogger("generative_data_prep_logger")


def combine_input_dir_files(input_path: str) -> Tuple[str, List[Path]]:
    """Processes a directory containing JSONL files and combines them into a single output file.

    Args:
        input_path (str): The path to the directory containing the input JSONL files.

    Returns:
        Tuple[str, str]: The string path to the combined output JSONL file. If there is only one JSONL file
            in the directory, returns the path to that file directly without combining. As well as all the
            input files.
    """
    input_path_obj = Path(input_path)

    if not input_path_obj.is_dir():
        raise ValueError(f"Input to combine_input_dir_files is not a valid directory: {input_path}")

    jsonl_files = list(input_path_obj.glob("*.jsonl"))
    txt_files = list(input_path_obj.glob("*.txt"))

    if jsonl_files:
        input_files = jsonl_files
        ext = ".jsonl"
    elif txt_files:
        input_files = txt_files
        ext = ".txt"
    else:
        raise ValueError(f"Invalid input path argument: {input_path}. No JSONL or TXT files found.")

    # If there's only one file, return it directly
    if len(input_files) == 1:
        return str(input_files[0]), input_files

    # Define the output path for the combined file
    output_file = input_path_obj / f"combined_output_{uuid.uuid4().hex[:8]}{ext}"

    # Open the output file and concatenate all input files
    with open(output_file, "w") as f_out:
        for input_file in input_files:
            if "combined_output_" not in str(input_file):
                verify_input_file(str(input_file))
                with open(input_file, "r") as f_in:
                    if input_file.stat().st_size == 0:
                        continue  # Skip empty files

                    shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)  # 8MB buffer

    return str(output_file), input_files


def split_file_linux(num_splits: int, input_file_path: str, split_dir: str) -> None:
    """Split the [input_file_path] into num_splits and places it in [split_dir].

    Args:
        num_splits (int): number of output file splits
        input_file_path (str): input jsonl file path
        split_dir (str): The directory to place all the outputted splits
    """
    split_command = f"split -d -n r/{num_splits} {input_file_path} {split_dir}/"
    execute_and_return_stdout(split_command)


def check_RAM(input_file_size_in_bytes: int):
    """Check to make sure there is enough RAM on the system to fit [input_file_size_in_bytes].

    Args:
        input_file_size_in_bytes: number of bytes in input file
    """
    available_RAM_in_bytes = psutil.virtual_memory().available
    shuffle_on_RAM = available_RAM_in_bytes > input_file_size_in_bytes
    err_msg = "you specified --shuffle=on_RAM, but there is not enough space on RAM to shuffle your file, available"
    err_msg += f"_RAM_in_bytes: {available_RAM_in_bytes} < input_file_size_in_bytes: {input_file_size_in_bytes}"
    if not shuffle_on_RAM:
        raise MemoryError("Not enough memory to shuffle load data onto RAM")


def rename_files(
    input_file_path: str,
    split_dir: str,
    train_count: int,
    dev_count: int,
    test_count: int,
    num_splits: int,
    test_dir: str,
    overwrite_output_path: bool,
) -> List[str]:
    """Take all the files in [split_dir] and renames them.

    Rename [train_count] of them to have train in the name, [dev_count] of them to have de in the name
    and places [test_count] of them into the test_dir

    Args:
        input_file_path: path to the input file
        split_dir: input directory that contains split files
        train_count: number of files to rename with train
        dev_count: number of files to rename with dev
        test_count: number of times to place into test directory
        test_dir: directory to place test files
        num_splits: number of splits that are in [split_dir]
        overwrite_output_path: If we can overwrite files
    """
    file_ext = os.path.splitext(input_file_path)[1]
    # rename the files to include 'train' and 'test'
    files_to_tokenize = []
    num_digits = len(str(num_splits))
    for i in range(num_splits):
        if i < train_count:
            new_name = f"train_{i+1}_of_{train_count}{file_ext}"
        elif i < train_count + test_count:
            new_name = f"test_{i-train_count+1}_of_{test_count}{file_ext}"
        else:
            new_name = f"dev_{i-train_count-test_count+1}_of_{dev_count}{file_ext}"

        new_file_path = os.path.join(split_dir, new_name)

        if os.path.exists(new_file_path) and not overwrite_output_path:
            err_msg = f"{new_file_path} already exists, and you are trying to overwrite it."
            err_msg += " To fix this error either specify --overwrite_output_path or move the conflicting file"
            raise ValueError(err_msg)

        os.rename(os.path.join(split_dir, str(i).zfill(max(2, num_digits))), new_file_path)
        if train_count <= i < train_count + test_count:
            os.rename(os.path.join(split_dir, new_name), os.path.join(test_dir, new_name))
        else:
            files_to_tokenize.append(new_name)

        if os.path.exists(new_file_path) and os.path.getsize(new_file_path) <= 0:
            raise ValueError(
                """The number of total splits exceeds the number of
        lines in the input path jsonl file. Please reduce the number
        of splits, or increase the number of lines in the dataset."""
            )
    return files_to_tokenize


def estimate_total_num_articles(files_to_tokenize, split_dir):
    """Estimates the total number of articles based on number of artiles in first split times number of splits.

    Args:
        files_to_tokenize: List of files to tokenize.
        split_dir: Directory where the split files are located.

    Returns:
        Estimate of the total number of articles needed to tokenize
    """
    lines_per_file = 0
    with open(os.path.join(split_dir, files_to_tokenize[0]), "r") as file:
        for _ in file:
            lines_per_file += 1

    return lines_per_file * len(files_to_tokenize)


def get_split_counts(
    input_file_size_in_gb: float,
    num_training_splits: Optional[int],
    num_dev_splits: Optional[int],
    num_test_splits: Optional[int],
    dev_ratio: Optional[float],
    test_ratio: Optional[float],
) -> Tuple[int, int, int, int]:
    """Based on the input arguments, returns the number number of output files to split into train, dev and test.

    If the splits are specified directly in the arguments, they are returned.
    If the ratios are specified the number of splits are calculated using num_training_splits.
    Only specify one of the two options
        num_training_splits and num_dev_splits and num_test_splits
            or
        num_training_splits and dev_ratio and test_ratio


    Args:
        input_file_size_in_gb: the size of the input file in gigabytes
        num_training_splits: number of training splits
        num_dev_splits: number of dev splits
        num_test_splits: number of test splits
        dev_ratio: ratio of dev splits
        test_ratio: ratio of test splits

    Returns:
        train_count, dev_count, test_count, num_splits
    """
    if num_training_splits is not None and num_test_splits is not None and num_dev_splits is not None:
        if test_ratio is not None:
            raise ValueError("you included the flag num_test_splits, so you can not specify the flag --test_ratio")
        if dev_ratio is not None:
            raise ValueError("you included the flag num_dev_splits, so you can not specify the flag --dev_ratio")
        train_count = num_training_splits
        test_count = num_test_splits
        dev_count = num_dev_splits
        num_splits = train_count + test_count + dev_count
    else:
        if num_test_splits is not None:
            err_msg = "You included the flag --num_test_splits, but did not include --num_dev_splits, or"
            err_msg += " --num_training_splits. If you want to use any of these flags, you must include all of them."
            raise ValueError(err_msg)
        if num_dev_splits is not None:
            err_msg = "You included the flag --num_dev_splits, but did not include --num_training_splits, "
            err_msg += "or --num_test_splits. If you want to use any of these flags, you must include all of them."
            raise ValueError(err_msg)

        dev_ratio = dev_ratio if dev_ratio is not None else 0.0
        test_ratio = test_ratio if test_ratio is not None else 0.0

        # determine number of train and test files
        train_count = get_num_training_splits(input_file_size_in_gb, num_training_splits)

        num_splits = int(train_count / (1 - dev_ratio - test_ratio))
        test_count = int(num_splits * test_ratio)
        dev_count = num_splits - test_count - train_count

    return train_count, dev_count, test_count, num_splits


def update_dataset_metadata(metrics: Metrics, dataset_metadata_json):
    """Update dataset metadata with prefixed or non-prefixed metric names."""
    if not metrics.is_empty:
        prefix = f"{metrics.dataset_type}_" if metrics.dataset_type else ""
        for key, value in vars(metrics).items():
            if key != "dataset_type":
                dataset_metadata_json.update({f"{prefix}{key}": value})


def data_prep_main_helper(args: Iterable[Any]):
    """Helper function to apply the star operator on the arguments when calling the data_prep_main function."""
    return data_prep_main(*args)


def multiprocess_data_prep(  # noqa: C901
    files_to_tokenize: List[str],
    split_dir: str,
    hdf5_dir: str,
    json_error_log_dir: str,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    prompt_keyword: str,
    completion_keyword: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    ignore_input_format_error: bool,
    tokenizer: PreTrainedTokenizerBase,
    num_workers: int,
    input_file_size_in_gb: float,
    dataset_metadata_json: Optional[Dict[str, Union[str, int, bool, None]]] = None,
    category_to_id: Optional[Dict[str, int]] = None,
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
    apply_chat_template: Optional[bool] = False,
) -> Tuple[List[str], List[str], Metrics, Metrics]:
    """Tokenizes all the files in files_to_tokenize efficiently using multirpocessing library.

    Args:
        files_to_tokenize: List of files to tokenize.
        split_dir: Directory that contains the files to tokenize.
        hdf5_dir: Directory to output tokenized hdf5 files.
        max_seq_length: Maximum sequence length of the model.
        input_packing_config: Packing style used during tokenization.
        packing_boundary: How to define the boundary when packing tokens.
        attention_boundary: How to define the boundary when attending to tokens.
        prompt_keyword: The keyword used to extract prompt from jsonl.
        completion_keyword: The keyword used to extract completion from jsonl.
        disable_space_separator: If true do not add space separators.
        keep_prompt_only_sequences: If true does not drop prompt-only sequences.
        tokenizer: The tokenizer to use for tokenizing text.
        num_workers: Number of workers to use for multiprocessing
        input_file_size_in_gb: Size of the input file in gigabytes.
        category_to_id: Dictionary that maps category names to ids.
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.

    Returns:
        List of output training and dev hdf5 file paths, and the metrics associated with tokenization
    """
    if input_file_size_in_gb > 10:
        log_sep_str()
        warning_msg = f"WARNING: your input file size is {input_file_size_in_gb} GB, "
        warning_msg += "this is large and may take up a lot of your machines resources for a long time."
        LOGGER.warning(warning_msg)
    log_sep_str()
    LOGGER.info(f"Running tokenization jobs locally, There are {num_workers} processes working on it.")
    sub_input_file_paths = list(map(lambda file_name: os.path.join(split_dir, file_name), files_to_tokenize))
    sub_output_file_paths = list(
        map(
            lambda file_name: os.path.join(hdf5_dir, f"{os.path.splitext(file_name)[0]}.hdf5"),
            files_to_tokenize,
        )
    )
    train_hdf5_files = list(filter(lambda file_name: "train" in file_name, sub_output_file_paths))
    dev_hdf5_files = list(filter(lambda file_name: "dev" in file_name, sub_output_file_paths))
    total_num_articles = estimate_total_num_articles(files_to_tokenize, split_dir)
    # create manager for shared variables to keep track of tokenization progress
    manager = multiprocessing.Manager()
    num_tokenized_articles_lock = manager.Lock()
    num_tokenized_articles = manager.Value(int, 0)
    num_skipped_articles = manager.Value(int, 0)
    prev_num_tokenized_articles = 0
    prev_num_skipped_articles = 0
    # Submit multiprocessing workers
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for input_file_path, output_file_path in zip(sub_input_file_paths, sub_output_file_paths):
        dataset_type = None
        if output_file_path in train_hdf5_files:
            dataset_type = "train"
        elif output_file_path in dev_hdf5_files:
            dataset_type = "dev"
        futures.append(
            executor.submit(
                data_prep_main_helper,
                (
                    True,
                    tokenizer,
                    input_file_path,
                    output_file_path,
                    json_error_log_dir,
                    max_seq_length,
                    input_packing_config,
                    packing_boundary,
                    attention_boundary,
                    disable_space_separator,
                    keep_prompt_only_sequences,
                    ignore_input_format_error,
                    prompt_keyword,
                    completion_keyword,
                    num_skipped_articles,
                    num_tokenized_articles,
                    num_tokenized_articles_lock,
                    category_to_id,
                    prompt_prefix,
                    prompt_postfix,
                    dataset_type,
                    apply_chat_template,
                ),
            )
        )

    broken_process_indices = []
    broken_process_pool_exc: Optional[BaseException] = None
    train_metrics = Metrics("train")
    dev_metrics = Metrics("dev")
    max_batch_size_train = None
    max_batch_size_dev = None
    tokenization_start_time = time.time()
    finished_futures = set()
    # Loop while processes are running, update progress bar.
    with alive_bar(total_num_articles) as bar:
        while True:
            for i, future in enumerate(futures):
                if future.done() and future not in finished_futures:
                    try:
                        indiv_metric = future.result()
                        if indiv_metric.dataset_type == "train":
                            if max_batch_size_train is None:
                                max_batch_size_train = indiv_metric.sequences
                            else:
                                max_batch_size_train = min(max_batch_size_train, indiv_metric.sequences)
                            train_metrics += indiv_metric
                        elif indiv_metric.dataset_type == "dev":
                            if max_batch_size_dev is None:
                                max_batch_size_dev = indiv_metric.sequences
                            else:
                                max_batch_size_dev = min(max_batch_size_dev, indiv_metric.sequences)
                            dev_metrics += indiv_metric
                        finished_futures.add(future)
                    except Exception as exc:
                        if isinstance(exc, concurrent.futures.process.BrokenProcessPool):
                            broken_process_indices.append(str(i))
                            broken_process_pool_exc = exc
                        else:
                            # If any process fails with NOT a BrokenProcessPool, show this error instead.
                            log_sep_str()
                            err_msg_1 = f"Process {i} failed with the exception below."
                            err_msg_2 = (
                                "If the error is a MemoryError, reduce the number of workers to limit your RAM usage."
                            )
                            LOGGER.error(f"\n\n{err_msg_1}\n{err_msg_2}")
                            raise exc from None
                        # if no "interesting" exceptions are found, raise the BrokenProcessPool Exception
                        if len(broken_process_indices) > 0:
                            log_sep_str()
                            LOGGER.error(
                                f'\n\nProcesses {", ".join(broken_process_indices)} failed with the exception:'
                            )
                            assert broken_process_pool_exc is not None  # nosec: B101
                            raise broken_process_pool_exc from None
            # If all the processes are done, break the loop
            if all(future.done() for future in futures):
                if len(finished_futures) != len(futures):
                    raise ValueError("All futures done, but finished futures set does not equal all futures list.")
                break
            # Update the progress bar with how every many new articles were tokenized
            with num_tokenized_articles_lock:
                num_new_tokenized_articles = num_tokenized_articles.value - prev_num_tokenized_articles
                bar(num_new_tokenized_articles)
                perc_complete = round((bar.current / total_num_articles) * 100, 2)
                elapsed_time_str = f"--- elapsed time: {time.time() - tokenization_start_time}"
                LOGGER.debug(
                    f"{total_num_articles}, {perc_complete}% complete => Time remaining: {bar.eta} {elapsed_time_str}"
                )
                prev_num_tokenized_articles = num_tokenized_articles.value

                if ignore_input_format_error:
                    num_new_skipped_articles = num_skipped_articles.value - prev_num_skipped_articles
                    if num_new_skipped_articles > 0:
                        LOGGER.info(f"{num_skipped_articles.value} misformatted lines are skipped")
                        prev_num_skipped_articles = num_skipped_articles.value
            time.sleep(5)

    if ignore_input_format_error:
        LOGGER.info(f"Total processed lines: {num_tokenized_articles.value}")
        LOGGER.info(f"Total skipped lines: {num_skipped_articles.value}")

    if dataset_metadata_json is not None:
        dataset_metadata_json["max_batch_size_train"] = max_batch_size_train
        dataset_metadata_json["max_batch_size_dev"] = max_batch_size_dev

    executor.shutdown()
    manager.shutdown()

    return train_hdf5_files, dev_hdf5_files, train_metrics, dev_metrics


def pipeline_main(  # noqa: C901
    input_path: str,
    tokenizer: PreTrainedTokenizerBase,
    pretrained_tokenizer: Optional[str],
    model_config: PretrainedConfig,
    output_dir: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    ignore_input_format_error: bool,
    prompt_keyword: str,
    completion_keyword: str,
    shuffle: str,
    overwrite_output_path: bool,
    num_workers: int,
    do_not_balance_hdf5: bool,
    keep_split_jsonls: bool,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    num_training_splits: Optional[int],
    num_dev_splits: Optional[int],
    num_test_splits: Optional[int],
    dev_ratio: Optional[float],
    test_ratio: Optional[float],
    category_to_id: Optional[Dict[str, int]] = None,
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
    apply_chat_template: Optional[bool] = False,
):
    """Endpoint for preparing data, shuffles, splits and tokenize input file.

    Args:
        input_path: Input file path of text to tokenize.
        tokenizer: Tokenizer used to tokenize text, with encode function.
        output_dir: Directory to output all the tokenized hdf5 and logs.
        disable_space_separator: If true do not add spaces between prompt and completion.
        keep_prompt_only_sequences: If true does not drop sequences that only have prompt tokens.
        prompt_keyword: The keyword used to extract prompt from jsonl.
        completion_keyword: The keyword used to extract completion from jsonl.
        shuffle: What kind of shuffling to perform, from [on_RAM, large_file, False]
        overwrite_output_path: Whether the output path should be deleted and over-written
        num_workers: Number of workers to use for multiprocessing
        do_not_balance_hdf5: If true, do not re-balance hdf5 files.
        keep_split_jsonls: If true, do not delete split jsonl files.
        max_seq_length: Maximum sequence length of the model.
        input_packing_config: Packing style used during tokenization.
        packing_boundary: How to define the boundary when packing text.
        attention_boundary: How to define the boundary of what tokens are attended to.
        split_dir: input directory that contains split files
        train_count: number of files to rename with train
        dev_count: number of files to rename with dev
        test_count: number of times to place into test directory
        test_dir: directory to place test files
        num_splits: number of splits that are in [split_dir]
        overwrite_output_path: If we can overwrite files

        num_training_splits: Number of training splits to create.
        num_dev_splits: Number of dev (evaluation) splits to create.
        num_test_splits: Number of test splits to create.
        dev_ratio: Ratio of data to use for dev (evaluation).
        test_ratio: Ratio of data to use as test.
        category_to_id: Dictionary that maps category string names to IDs.
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.

    Raises:
        RuntimeError: If shuffling on RAM is not possible

    Returns:
        Metrics associated with tokenization, Dataset metadata
    """
    input_file_path = input_path
    if os.path.isdir(input_path):
        input_file_path, input_files = combine_input_dir_files(input_path)

    # print input file information
    dataset_metadata_json = {
        "max_seq_length": max_seq_length,
        "token_type_ids": True,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_model_type": str(type(model_config)),
    }
    input_file_size_in_bytes = os.stat(input_file_path).st_size
    input_file_size_in_gb = input_file_size_in_bytes / (1024**3)
    log_message = f"Size of input jsonl file is: {round(input_file_size_in_gb, 2)} GB"
    log_message += f" ({round(input_file_size_in_bytes / (1024**2), 2)} MB)"
    log_sep_str()
    LOGGER.info(log_message)
    if input_file_size_in_bytes <= 1:
        raise ValueError(f"your inputted file {input_file_path} is empty")

    train_count, dev_count, test_count, num_splits = get_split_counts(
        input_file_size_in_gb,
        num_training_splits,
        num_dev_splits,
        num_test_splits,
        dev_ratio,
        test_ratio,
    )

    num_splits_greater_lines = False
    with open(input_file_path, "r") as input_file:
        for i, line in enumerate(input_file):
            if i > num_splits:
                num_splits_greater_lines = True
                break
    if not num_splits_greater_lines:
        raise ValueError(
            """The number of total splits exceeds the number of
        lines in the input path jsonl file. Please reduce the number
        of splits, or increase the number of lines in the dataset."""
        )
    dataset_metadata_json["number_of_training_files"] = train_count
    dataset_metadata_json["number_of_dev_files"] = dev_count
    dataset_metadata_json["number_of_test_files"] = test_count

    split_dir = os.path.join(output_dir, "splits")
    verify_output_dir(split_dir, False)

    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    verify_output_dir(tokenizer_dir, True)
    save_tokenizer(tokenizer, tokenizer_dir, pretrained_tokenizer)

    model_config_path = os.path.join(tokenizer_dir, "config.json")
    model_config.to_json_file(model_config_path)

    json_error_log_dir = os.path.join(output_dir, "json_error_log")
    verify_output_dir(json_error_log_dir, True)

    if category_to_id is not None:
        category_to_id_output_file_path = os.path.join(output_dir, "category_to_id.json")
        verify_output_file(category_to_id_output_file_path, overwrite_output_path)
        with open(category_to_id_output_file_path, "w") as f:
            json.dump(category_to_id, f)

    test_dir = os.path.join(output_dir, "test_files")
    if test_count > 0:
        verify_output_dir(test_dir, False)

    # Shuffle and split the input file
    # =========================================================
    # Case 1: large file shuffle specified. REQUIRES: linux OS
    if shuffle == "large_file":
        err_msg = "You specified --shuffle=large_file, but this is only supported on linux operating systems, "
        err_msg += f"your operating system is {platform}. Please change the flag to --shuffle=on_RAM or --shuffle=False"
        if "linux" not in platform.lower():
            raise OSError(err_msg)
        split_dir = large_file_shuffle(input_file_path, output_dir, False, num_splits)

    # Case 2: Shuffling on RAM with linux OS
    elif shuffle == "on_RAM" and "linux" in platform.lower():
        check_RAM(input_file_size_in_bytes)
        log_sep_str()
        LOGGER.info("Shuffling input file, please be patient.")
        file_ext = os.path.splitext(input_file_path)[1]
        shuffle_file_path = os.path.join(output_dir, f"tmp_shuf{file_ext}")
        shuffle_command = f"shuf {input_file_path} > {shuffle_file_path}"
        try:
            out = execute_and_return_stdout(shuffle_command)
            err_msg = f"Shuffle command killed, with print stdout:{out.stdout} stderr:{out.stderr}"
            if "killed" in out.stdout or "killed" in out.stderr:
                raise MemoryError(err_msg)
        except Exception as e:
            err_msg = f"Failed with exception {e}, shuffling on RAM is not possible,"
            err_msg += " try specifying argument --shuffle=large_file"
            raise RuntimeError(err_msg)
        split_file_linux(num_splits, shuffle_file_path, split_dir)
        os.remove(shuffle_file_path)

    # Case 3: shuffle on RAM without linux OS
    elif shuffle == "on_RAM" and "linux" not in platform.lower():
        check_RAM(input_file_size_in_bytes)
        lines = open(input_file_path).readlines()
        random.shuffle(lines)
        splits = np.array_split(lines, num_splits)
        num_digits = len(str(num_splits))
        for i, split in enumerate(splits):
            out_file_path = os.path.join(split_dir, str(i).zfill(max(2, num_digits)))
            with open(out_file_path, "w") as out_file:
                out_file.writelines(split)

    # Case 4: Do not shuffle, split file without linux OS
    elif shuffle == "False" and "linux" not in platform.lower():
        log_sep_str()
        LOGGER.warning("WARNING: you did not specify the --shuffle flag, so no shuffling was done!")
        out_files = []
        num_digits = len(str(num_splits))
        for i in range(num_splits):
            out_file_path = os.path.join(split_dir, str(i).zfill(max(2, num_digits)))
            out_files.append(out_file_path)
            with open(out_file_path, "w") as _:
                pass

        with open(input_file_path, "r") as input_file:
            for i, line in enumerate(input_file):
                with open(out_files[i % len(out_files)], "a") as out_f:
                    out_f.write(line)

    # Case 5: Do not shuffle, split file with linux OS
    elif shuffle == "False" and "linux" in platform.lower():
        log_sep_str()
        LOGGER.warning("WARNING: you did not specify the --shuffle flag, so no shuffling was done!")
        split_file_linux(num_splits, input_file_path, split_dir)

    # rename files to include the corresponding names of 'test', 'dev' and 'train'
    files_to_tokenize = rename_files(
        input_file_path,
        split_dir,
        train_count,
        dev_count,
        test_count,
        num_splits,
        test_dir,
        overwrite_output_path,
    )

    train_hdf5_files, dev_hdf5_files, train_metrics, dev_metrics = multiprocess_data_prep(
        files_to_tokenize,
        split_dir,
        output_dir,
        json_error_log_dir,
        max_seq_length,
        input_packing_config,
        packing_boundary,
        attention_boundary,
        prompt_keyword,
        completion_keyword,
        disable_space_separator,
        keep_prompt_only_sequences,
        ignore_input_format_error,
        tokenizer,
        num_workers,
        input_file_size_in_gb,
        dataset_metadata_json,
        category_to_id,
        prompt_prefix,
        prompt_postfix,
        apply_chat_template,
    )

    log_sep_str()
    LOGGER.info(f"Tokenization is complete, the output dataset is located at: {output_dir}")

    # Balance hdf5 files so they all have the same number of sequences to within 1
    if do_not_balance_hdf5:
        log_sep_str()
        warning = "WARNING: Skipping balancing hdf5 files, this is not recommended because during "
        warning += 'distributed training some workers will train on some data more than once per "epoch".'
        LOGGER.warning(warning)

    else:
        log_sep_str()
        LOGGER.info("Balancing hdf5 files to ensure they have the same number of sequences.")
        balance_hdf5_files(train_hdf5_files, dataset_metadata_json, "train")
        balance_hdf5_files(dev_hdf5_files, dataset_metadata_json, "dev")

    if not keep_split_jsonls:
        shutil.rmtree(split_dir)

    file_names = []
    for file_name in os.listdir(json_error_log_dir):
        file_names.append(os.path.join(json_error_log_dir, file_name))
    if file_names:
        with open(os.path.join(output_dir, "json_load_failed_lines.log"), "w") as outfile:
            for file_name in file_names:
                with open(file_name) as reader:
                    for line in reader:
                        outfile.write(line)
    shutil.rmtree(json_error_log_dir)

    if os.path.isdir(input_path) and len(input_files) > 1:
        os.remove(input_file_path)

    update_dataset_metadata(train_metrics, dataset_metadata_json)
    update_dataset_metadata(dev_metrics, dataset_metadata_json)
    metadata_file_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file_path, "w") as file:
        yaml.dump(dataset_metadata_json, file, default_flow_style=False)

    # Create sha256 of all the files within the directory
    create_sha256(output_dir)

    return train_metrics, dev_metrics, dataset_metadata_json
