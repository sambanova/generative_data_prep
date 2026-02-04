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
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import psutil
import yaml
from alive_progress import alive_bar
from transformers import PretrainedConfig, PreTrainedTokenizerBase

# Set multiprocessing start method for Windows compatibility
if sys.platform == "win32":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass
from generative_data_prep.data_prep import data_prep_main
from generative_data_prep.processors.metrics import Metrics
from generative_data_prep.utils import (
    BoundaryType,
    PackingConfig,
    balance_hdf5_files,
    create_sha256,
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
    with open(output_file, "w", encoding="utf-8", errors="replace") as f_out:
        for input_file in input_files:
            if "combined_output_" not in str(input_file):
                verify_input_file(str(input_file))
                with open(input_file, "r", encoding="utf-8", errors="replace") as f_in:
                    if input_file.stat().st_size == 0:
                        continue  # Skip empty files

                    shutil.copyfileobj(f_in, f_out, length=8 * 1024 * 1024)  # 8MB buffer

    return str(output_file), input_files


def split_file_round_robin(num_splits: int, input_file_path: str, split_dir: str) -> None:
    """Split the [input_file_path] into num_splits and places it in [split_dir] using round-robin distribution.

    This is a cross-platform replacement for the Linux 'split -d -n r/' command.

    Args:
        num_splits (int): number of output file splits
        input_file_path (str): input jsonl file path
        split_dir (str): The directory to place all the outputted splits
    """
    # Create file handles for all split files
    split_files = []
    num_digits = len(str(num_splits))
    for i in range(num_splits):
        out_file_path = os.path.join(split_dir, str(i).zfill(max(2, num_digits)))
        split_files.append(open(out_file_path, "w", encoding="utf-8", errors="replace"))

    try:
        # Read input file and distribute lines in round-robin fashion
        with open(input_file_path, "r", encoding="utf-8", errors="replace") as infile:
            for line_num, line in enumerate(infile):
                split_index = line_num % num_splits
                split_files[split_index].write(line)
    finally:
        # Close all file handles
        for f in split_files:
            f.close()


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
            new_name = f"train_{i + 1}_of_{train_count}{file_ext}"
        elif i < train_count + test_count:
            new_name = f"test_{i - train_count + 1}_of_{test_count}{file_ext}"
        else:
            new_name = f"dev_{i - train_count - test_count + 1}_of_{dev_count}{file_ext}"

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


def count_exact_total_num_articles(files_to_tokenize, split_dir):
    """Counts the exact total number of articles by counting all non-empty lines in all files.

    Args:
        files_to_tokenize: List of files to tokenize.
        split_dir: Directory where the split files are located.

    Returns:
        Exact count of the total number of articles to tokenize
    """
    if not files_to_tokenize:
        return 0

    total_lines = 0
    LOGGER.info(f"Counting articles in {len(files_to_tokenize)} files to get exact total...")

    for file_name in files_to_tokenize:
        file_path = os.path.join(split_dir, file_name)
        lines_in_file = 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                # Skip empty lines to match actual processing behavior
                if line.strip():
                    lines_in_file += 1
        total_lines += lines_in_file

    LOGGER.info(f"Exact total articles counted: {total_lines}")
    return total_lines


def estimate_total_num_articles(files_to_tokenize, split_dir):
    """Estimates the total number of articles based on number of articles in sample files times number of splits.

    DEPRECATED: Use count_exact_total_num_articles for exact count instead.

    Args:
        files_to_tokenize: List of files to tokenize.
        split_dir: Directory where the split files are located.

    Returns:
        Estimate of the total number of articles needed to tokenize
    """
    if not files_to_tokenize:
        return 0

    # Sample up to 5 files to get a better average estimate
    sample_size = min(5, len(files_to_tokenize))
    total_lines = 0
    files_sampled = 0

    for i in range(sample_size):
        file_path = os.path.join(split_dir, files_to_tokenize[i])
        lines_in_file = 0
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                # Skip empty lines to match actual processing behavior
                if line.strip():
                    lines_in_file += 1
        total_lines += lines_in_file
        files_sampled += 1

    if files_sampled == 0:
        return 0

    # Calculate average lines per file and multiply by total files
    avg_lines_per_file = total_lines / files_sampled
    return int(avg_lines_per_file * len(files_to_tokenize))


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
    # Count exact total to guarantee 100% accuracy
    total_num_articles = count_exact_total_num_articles(files_to_tokenize, split_dir)
    # create manager for shared variables to keep track of tokenization progress
    manager = multiprocessing.Manager()
    num_tokenized_articles_lock = manager.Lock()
    num_tokenized_articles = manager.Value(int, 0)
    num_skipped_articles = manager.Value(int, 0)
    prev_num_tokenized_articles = 0
    prev_num_skipped_articles = 0
    # Track how much we've actually updated the progress bar to prevent exceeding total
    bar_update_tracker = 0
    # Submit multiprocessing workers
    # On Windows, reduce workers to avoid pickling issues with large tokenizers
    if sys.platform == "win32" and num_workers > 4:
        LOGGER.warning(f"Reducing workers from {num_workers} to 4 on Windows to avoid multiprocessing issues.")
        num_workers = 4
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
    # Use manual mode to have better control over the progress bar
    with alive_bar(total_num_articles, manual=True, title="Tokenizing articles") as bar:
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
                # Final update to ensure progress bar reflects all processed articles
                with num_tokenized_articles_lock:
                    num_new_tokenized_articles = num_tokenized_articles.value - prev_num_tokenized_articles
                    if num_new_tokenized_articles > 0:
                        # Use our tracker to ensure we never exceed total
                        remaining_until_total = max(0, total_num_articles - bar_update_tracker)
                        if remaining_until_total > 0:
                            # Cap update to not exceed total
                            max_update = min(num_new_tokenized_articles, remaining_until_total)
                            if max_update > 0:
                                bar_update_tracker += max_update
                                # Set bar to exact position (as fraction of total, capped at 1.0)
                                bar_position = (
                                    min(1.0, bar_update_tracker / total_num_articles) if total_num_articles > 0 else 0.0
                                )
                                bar(bar_position)
                # Ensure progress bar reaches exactly 100% (1.0 in manual mode)
                # Use tracker to set final position
                bar_update_tracker = total_num_articles
                bar(1.0)  # Set to 100% completion
                break
            # Update the progress bar with how every many new articles were tokenized
            with num_tokenized_articles_lock:
                num_new_tokenized_articles = num_tokenized_articles.value - prev_num_tokenized_articles
                if num_new_tokenized_articles > 0:
                    # Use our tracker to ensure we never exceed total
                    remaining_until_total = max(0, total_num_articles - bar_update_tracker)
                    # Only update if there's room and we have new articles
                    if remaining_until_total > 0:
                        # Cap update to not exceed total
                        max_update = min(num_new_tokenized_articles, remaining_until_total)
                        if max_update > 0:
                            bar_update_tracker += max_update
                            # Set bar to exact position (as fraction of total, capped at 1.0)
                            bar_position = (
                                min(1.0, bar_update_tracker / total_num_articles) if total_num_articles > 0 else 0.0
                            )
                            bar(bar_position)
                # Calculate percentage based on our tracker (more accurate than bar.current in manual mode)
                if total_num_articles > 0:
                    # Use tracker to calculate accurate percentage
                    actual_current = min(bar_update_tracker, total_num_articles)
                    perc_complete = min(100.0, round((actual_current / total_num_articles) * 100, 2))
                else:
                    perc_complete = 0.0
                elapsed_time_str = f"--- elapsed time: {time.time() - tokenization_start_time}"
                LOGGER.debug(
                    f"Counter: {num_tokenized_articles.value}, Progress tracker: "
                    f"{bar_update_tracker}/{total_num_articles}, {perc_complete}% complete => "
                    f"Time remaining: {bar.eta} {elapsed_time_str}"
                )
                prev_num_tokenized_articles = num_tokenized_articles.value

                if ignore_input_format_error:
                    num_new_skipped_articles = num_skipped_articles.value - prev_num_skipped_articles
                    if num_new_skipped_articles > 0:
                        LOGGER.info(f"{num_skipped_articles.value} misformatted lines are skipped")
                        prev_num_skipped_articles = num_skipped_articles.value
            time.sleep(5)

    # Log final article count and validate 100% completion
    log_sep_str()
    total_actual_articles = train_metrics.articles + dev_metrics.articles
    LOGGER.info(
        f"Total articles processed (from metrics): {total_actual_articles} "
        f"(Train: {train_metrics.articles}, Dev: {dev_metrics.articles})"
    )
    LOGGER.info(f"Total articles counted in input files: {total_num_articles}")

    if ignore_input_format_error:
        LOGGER.info(f"Progress counter value: {num_tokenized_articles.value}")
        LOGGER.info(f"Total skipped lines (format errors): {num_skipped_articles.value}")

    # Validate 100% completion
    if total_num_articles > 0:
        counter_articles = num_tokenized_articles.value
        metrics_articles = total_actual_articles
        skipped_articles = num_skipped_articles.value if ignore_input_format_error else 0

        # Calculate expected articles (total - skipped due to format errors)
        # Note: Articles dropped during processing (prompt-only, packing drops)
        # are still counted in metrics.articles because metrics.articles is
        # incremented before processing/dropping
        expected_articles = total_num_articles - skipped_articles

        # Compare metrics with expected count
        metrics_diff = abs(metrics_articles - expected_articles)
        metrics_diff_percent = (metrics_diff / total_num_articles) * 100 if total_num_articles > 0 else 0

        log_sep_str()
        if metrics_diff == 0:
            LOGGER.info(
                f"[SUCCESS] 100% DATA UTILIZATION: All {total_num_articles} "
                f"articles from input files were processed!"
            )
            if skipped_articles > 0:
                LOGGER.info(
                    f"  Note: {skipped_articles} articles were skipped due to " f"JSON format errors (expected)"
                )
            LOGGER.info(f"  All {metrics_articles} processed articles are included in " f"the output dataset.")
        elif metrics_diff_percent <= 0.1:  # Less than 0.1% difference
            LOGGER.warning(
                f"Near-complete data utilization: {metrics_articles}/"
                f"{expected_articles} articles processed "
                f"({metrics_diff_percent:.3f}% difference). This is likely due "
                f"to rounding or minor counting differences."
            )
            LOGGER.info(f"  {metrics_articles} articles are included in the output " f"dataset.")
        else:
            LOGGER.error(
                f"[WARNING] INCOMPLETE DATA UTILIZATION: Only "
                f"{metrics_articles}/{expected_articles} articles processed "
                f"({metrics_diff_percent:.2f}% difference, "
                f"{expected_articles - metrics_articles} articles missing)."
            )
            LOGGER.error(
                f"  This means {expected_articles - metrics_articles} articles "
                f"from your input files were not processed. "
                f"Please check for errors in processing or data format issues."
            )

        # Compare counter with metrics to identify counting issues
        if abs(counter_articles - metrics_articles) > 10:
            LOGGER.warning(
                f"Counter discrepancy detected: Progress counter shows "
                f"{counter_articles} articles, but metrics show "
                f"{metrics_articles} articles were actually processed. "
                f"Difference: {abs(metrics_articles - counter_articles)} articles. "
                f"The metrics count ({metrics_articles}) is the accurate one."
            )
        else:
            LOGGER.info(
                f"[OK] Progress counter matches metrics: {counter_articles} "
                f"articles counted, {metrics_articles} articles processed."
            )

        log_sep_str()

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
    with open(input_file_path, "r", encoding="utf-8", errors="replace") as input_file:
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
        with open(category_to_id_output_file_path, "w", encoding="utf-8") as f:
            json.dump(category_to_id, f)

    test_dir = os.path.join(output_dir, "test_files")
    if test_count > 0:
        verify_output_dir(test_dir, False)

    # Shuffle and split the input file
    # =========================================================
    # Case 1: large file shuffle specified. REQUIRES: linux OS
    if shuffle == "large_file":
        split_dir = large_file_shuffle(input_file_path, output_dir, False, num_splits)

    # Case 2: Shuffling on RAM (cross-platform)
    elif shuffle == "on_RAM":
        check_RAM(input_file_size_in_bytes)
        log_sep_str()
        LOGGER.info("Shuffling input file, please be patient.")
        # Read all lines into memory
        with open(input_file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # Shuffle the lines
        random.shuffle(lines)
        # Split into chunks
        splits = np.array_split(lines, num_splits)
        num_digits = len(str(num_splits))
        for i, split in enumerate(splits):
            out_file_path = os.path.join(split_dir, str(i).zfill(max(2, num_digits)))
            with open(out_file_path, "w", encoding="utf-8", errors="replace") as out_file:
                out_file.writelines(split)

    # Case 3: Do not shuffle, split file (cross-platform)
    elif shuffle == "False":
        log_sep_str()
        LOGGER.warning("WARNING: you did not specify the --shuffle flag, so no shuffling was done!")
        split_file_round_robin(num_splits, input_file_path, split_dir)

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
        with open(os.path.join(output_dir, "json_load_failed_lines.log"), "w", encoding="utf-8") as outfile:
            for file_name in file_names:
                with open(file_name, "r", encoding="utf-8") as reader:
                    for line in reader:
                        outfile.write(line)
    shutil.rmtree(json_error_log_dir)

    if os.path.isdir(input_path) and len(input_files) > 1:
        os.remove(input_file_path)

    update_dataset_metadata(train_metrics, dataset_metadata_json)
    update_dataset_metadata(dev_metrics, dataset_metadata_json)
    metadata_file_path = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file_path, "w", encoding="utf-8") as file:
        yaml.dump(dataset_metadata_json, file, default_flow_style=False)

    # Create sha256 of all the files within the directory
    create_sha256(output_dir)

    return train_metrics, dev_metrics, dataset_metadata_json
