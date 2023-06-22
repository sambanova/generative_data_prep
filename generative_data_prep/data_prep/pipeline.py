"""
Copyright 2023 SambaNova Systems, Inc.

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

import os
import random
from multiprocessing import Pool
from sys import platform
from typing import List, Optional, Tuple

import numpy as np
import psutil
from transformers import PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main
from generative_data_prep.utils import (
    SEP_STR,
    BoundaryType,
    PackingConfig,
    balance_hdf5_files,
    execute_and_return_stdout,
    large_file_shuffle,
    verify_output_dir,
)


def split_file_linux(num_splits: int, input_file_path: str, split_dir: str) -> None:
    """Split the [input_file_path] into num_splits and places it in [split_dir].

    Args:
        num_splits (int): number of output file splits
        input_file_path (str): input jsonl file path
        split_dir (str): The directory to place all the outputted splits
    """
    split_command = f"split -d -n r/{num_splits} {input_file_path} {split_dir}/"
    os.system(split_command)


def check_RAM(input_file_size_in_bytes: int):
    """Check to make sure there is enough RAM on the system to fit [input_file_size_in_bytes].

    Args:
        input_file_size_in_bytes: number of bytes in input file
    """
    available_RAM_in_bytes = psutil.virtual_memory().available
    shuffle_on_RAM = available_RAM_in_bytes > input_file_size_in_bytes
    err_msg = "you specified --shuffle=on_RAM, but there is not enough space on RAM to shuffle your file, available"
    err_msg += f"_RAM_in_bytes: {available_RAM_in_bytes} < input_file_size_in_bytes: {input_file_size_in_bytes}"
    assert shuffle_on_RAM, err_msg


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

        err_msg = f"{new_file_path} already exists, and you are trying to overwrite it."
        err_msg += " To fix this error either specify --overwrite_output_path or move the conflicting file"
        assert not os.path.exists(new_file_path) or overwrite_output_path, err_msg

        os.rename(os.path.join(split_dir, str(i).zfill(max(2, num_digits))), new_file_path)
        if train_count <= i < train_count + test_count:
            os.rename(os.path.join(split_dir, new_name), os.path.join(test_dir, new_name))
        else:
            files_to_tokenize.append(new_name)
    return files_to_tokenize


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
        assert test_ratio is None, "you included the flag num_test_splits, so you can not specify the flag --test_ratio"
        assert dev_ratio is None, "you included the flag num_dev_splits, so you can not specify the flag --dev_ratio"
        train_count = num_training_splits
        test_count = num_test_splits
        dev_count = num_dev_splits
        num_splits = train_count + test_count + dev_count
    else:
        err_msg = "You included the flag --num_test_splits, but did not include --num_dev_splits, or"
        err_msg += " --num_training_splits. If you want to use any of these flags, you must include all of them."
        assert num_test_splits is None, err_msg
        err_msg = "You included the flag --num_dev_splits, but did not include --num_training_splits, "
        err_msg += "or --num_test_splits. If you want to use any of these flags, you must include all of them."
        assert num_dev_splits is None, err_msg

        dev_ratio = dev_ratio if dev_ratio is not None else 0.0
        test_ratio = test_ratio if test_ratio is not None else 0.0

        # determine number of train and test files
        if num_training_splits is None:
            # number of splits by default
            if input_file_size_in_gb < 10:
                train_count = 32
            elif input_file_size_in_gb < 100:
                train_count = 128
            else:
                train_count = 256
        else:
            train_count = num_training_splits

        num_splits = int(train_count / (1 - dev_ratio - test_ratio))
        test_count = int(num_splits * test_ratio)
        dev_count = num_splits - test_count - train_count

    return train_count, dev_count, test_count, num_splits


def multiprocess_data_prep(
    files_to_tokenize: List[str],
    split_dir: str,
    hdf5_dir: str,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    prompt_keyword: str,
    completion_keyword: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    tokenizer: PreTrainedTokenizerBase,
    num_workers: int,
    input_file_size_in_gb: float,
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
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
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.

    Returns:
        List of output training and dev hdf5 file paths
    """
    print(SEP_STR)
    print(f"Running tokenization jobs locally, There are {num_workers} processes working on it")
    if input_file_size_in_gb > 10:
        warning_msg = f"your input file size is {input_file_size_in_gb} GB, "
        warning_msg += "this is large and may take up a lot of your machines resources for a long time"
        print(warning_msg)
    sub_input_file_paths = list(map(lambda file_name: os.path.join(split_dir, file_name), files_to_tokenize))
    sub_output_file_paths = list(
        map(
            lambda file_name: os.path.join(hdf5_dir, f"{os.path.splitext(file_name)[0]}.hdf5"),
            files_to_tokenize,
        )
    )
    train_hdf5_files = list(filter(lambda file_name: "train" in file_name, sub_output_file_paths))
    dev_hdf5_files = list(filter(lambda file_name: "dev" in file_name, sub_output_file_paths))

    data_prep_main_args_list = []
    for input_file_path, output_file_path in zip(sub_input_file_paths, sub_output_file_paths):
        data_prep_main_args_list.append(
            (
                True,
                tokenizer,
                input_file_path,
                output_file_path,
                max_seq_length,
                input_packing_config,
                packing_boundary,
                attention_boundary,
                disable_space_separator,
                keep_prompt_only_sequences,
                prompt_keyword,
                completion_keyword,
                prompt_prefix,
                prompt_postfix,
            )
        )

    with Pool(num_workers) as p:
        _ = p.starmap(data_prep_main, data_prep_main_args_list)

    return train_hdf5_files, dev_hdf5_files


def pipeline_main(
    input_file_path: str,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    prompt_keyword: str,
    completion_keyword: str,
    shuffle: str,
    overwrite_output_path: bool,
    num_workers: int,
    do_not_balance_hdf5: bool,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    num_training_splits: Optional[int],
    num_dev_splits: Optional[int],
    num_test_splits: Optional[int],
    dev_ratio: Optional[float],
    test_ratio: Optional[float],
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
):
    """Endpoint for preparing data, shuffles, splits and tokenize input file.

    Args:
        input_file_path: Input file path of text to tokenize.
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
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.
    Raises:
        RuntimeError: If shuffling on RAM is not possible
    """
    # print input file information
    input_file_size_in_bytes = os.stat(input_file_path).st_size
    input_file_size_in_gb = input_file_size_in_bytes / (1024**3)
    print(SEP_STR)
    print(
        "Size of input jsonl file is: {:.2f} GB, or {:.2f} MB".format(
            input_file_size_in_gb, input_file_size_in_bytes / (1024**2)
        )
    )
    assert input_file_size_in_bytes > 1, f"your inputted file {input_file_path} is empty"

    train_count, dev_count, test_count, num_splits = get_split_counts(
        input_file_size_in_gb,
        num_training_splits,
        num_dev_splits,
        num_test_splits,
        dev_ratio,
        test_ratio,
    )

    split_dir = os.path.join(output_dir, "splits")
    verify_output_dir(split_dir, False)

    hdf5_dir = os.path.join(output_dir, "hdf5")
    verify_output_dir(hdf5_dir, False)

    tokenizer_dir = os.path.join(hdf5_dir, "tokenizer")
    verify_output_dir(tokenizer_dir, True)
    tokenizer.save_pretrained(tokenizer_dir)

    test_dir = os.path.join(output_dir, "test_files")
    if test_count > 0:
        verify_output_dir(test_dir, False)

    # Shuffle and split the input file
    # =========================================================
    # Case 1: large file shuffle specified. REQUIRES: linux OS
    if shuffle == "large_file":
        err_msg = "You specified --shuffle=large_file, but this is only supported on linux operating systems, "
        err_msg += f"your operating system is {platform}. Please change the flag to --shuffle=on_RAM or --shuffle=False"
        assert "linux" in platform.lower(), err_msg
        split_dir = large_file_shuffle(input_file_path, output_dir, False, num_splits)

    # Case 2: Shuffling on RAM with linux OS
    elif shuffle == "on_RAM" and "linux" in platform.lower():
        check_RAM(input_file_size_in_bytes)
        print(SEP_STR)
        print("shuffling input file, please be patient")
        file_ext = os.path.splitext(input_file_path)[1]
        shuffle_file_path = os.path.join(output_dir, f"tmp_shuf{file_ext}")
        shuffle_command = f"shuf {input_file_path} > {shuffle_file_path}"
        try:
            out = execute_and_return_stdout(shuffle_command)
            err_msg = f"Shuffle command killed, with print stdout:{out.stdout} stderr:{out.stderr}"
            assert "killed" not in out.stdout and "killed" not in out.stderr, err_msg
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
        print(SEP_STR)
        print("You did not specify the --shuffle flag, so no shuffling was done!")
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
        print(SEP_STR)
        print("You did not specify the --shuffle flag, so no shuffling was done!")
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

    train_hdf5_files, dev_hdf5_files = multiprocess_data_prep(
        files_to_tokenize,
        split_dir,
        hdf5_dir,
        max_seq_length,
        input_packing_config,
        packing_boundary,
        attention_boundary,
        prompt_keyword,
        completion_keyword,
        disable_space_separator,
        keep_prompt_only_sequences,
        tokenizer,
        num_workers,
        input_file_size_in_gb,
        prompt_prefix,
        prompt_postfix,
    )

    print(f"Tokenization is complete, the outputs are in {hdf5_dir}, the held out test files are located at {test_dir}")
    print(SEP_STR)

    # Balance hdf5 files so they all have the same number of sequences to within 1
    if do_not_balance_hdf5:
        warning = "WARNING: Skipping balancing hdf5 files, this is not recommended because during "
        warning += 'distributed training some workers will train on some data more than once per "epoch"'
        print(warning)
        print(SEP_STR)
    else:
        print("Balancing hdf5 files to ensure they have the same number of sequences")
        balance_hdf5_files(train_hdf5_files)
        balance_hdf5_files(dev_hdf5_files)
        print(f"Hdf5 balancing is complete, the outputs are located at {hdf5_dir}")
        print(SEP_STR)
