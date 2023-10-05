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


Entry point to the Text Processing Pipeline.
"""

import argparse
import json
import logging
import os
from multiprocessing import cpu_count
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main, pipeline_main
from generative_data_prep.utils import (
    GPT2_KEY,
    SEP_STR,
    TOKENIZER_CLASSES,
    FileExtension,
    data_prep_arg_builder,
    logger,
    verify_input_file,
    verify_output_dir,
    verify_output_file,
)


def add_data_prep_args(subparser: argparse.ArgumentParser):
    """Create the argparser for the generative_data_prep/data_prep/data_prep.py script.

    Args:
        subparser: The parser to add the arguments to.
    """
    subparser.add_argument(
        "--silent",
        default=False,
        action="store_true",
        required=False,
        help="Do not allow this script to print",
    )
    data_prep_arg_builder(subparser)


def add_pipeline_args(subparser: argparse.ArgumentParser):
    """Create the argparser for the generative_data_prep/data_prep/pipeline.py script.

    Args:
        subparser: The parser to add the arguments to
    """
    subparser.add_argument(
        "--num_training_splits",
        default=None,
        type=int,
        required=False,
        help="The number of training files to split input data into. If you specify the --dev_ratio and --test_ratio \
        flags, The total number of splits will be (num_training_splits / (1-dev_ratio-test_ratio)), and the number \
        of dev and test splits are calculated accordingly. If you specify --num_dev_splits and --num_test_splits \
        flags then those will  directory define the number of splits and therefore the ratios. We recommend you do \
        not include this flag and allow it to default, because the number of training splits must be greater than \
        the number of parallel workers and it is best if the number of training splits is a multiple of the number \
        of workers. It defaults to 32 training splits if input_file_size < 10GB, 128 training splits if 10GB < \
        input_file_size <100GB, 256 training splits if 100GB < input_file_size.",  # noqa: E251
    )
    subparser.add_argument(
        "--dev_ratio",
        default=None,
        type=float,
        required=False,
        help="The ratio of data that should be excluded from train set and used for evaluation, defaults to 10%%. If \
        you specify this flag, do not specify --num_dev_splits or --num_test_splits.",  # noqa: E251
    )
    subparser.add_argument(
        "--num_dev_splits",
        default=None,
        type=int,
        required=False,
        help="If you do not specify --dev_ratio, you may specify num_dev_splits. If you include this flag, you must \
        also include the --num_dev_splits and --num_training_splits flags",  # noqa: E251
    )
    subparser.add_argument(
        "--test_ratio",
        default=None,
        type=float,
        required=False,
        help="The ratio of data that should be excluded from train set and is saved for testing. This data is not \
        tokenized and left in jsonl format, defaults to 0%%. If you specify this flag, do not specify \
        --num_dev_splits or --num_test_splits.",  # noqa: E251
    )
    subparser.add_argument(
        "--num_test_splits",
        default=None,
        type=int,
        required=False,
        help="If you do not specify --test_ratio, you may specify num_dev_splits. If you include this flag, you must \
        also include the --num_dev_splits and --num_training_splits flags.",  # noqa: E251
    )
    subparser.add_argument(
        "--shuffle",
        default="False",
        const="False",
        nargs="?",
        choices=["False", "on_RAM", "large_file"],
        help="Choose the on_RAM option if your file is small enough to fit on RAM (If you are not sure if it fits \
        on RAM, default to this flag). If you are running a linux operating system and your file is too large to fit \
        on RAM, please choose large_file option, this will run approximate file shuffling that can handle files of \
        any size. If you want to do large file shuffling but you are not on linux, please shuffle the file before \
        using this script. If the input file should not be shuffled, do not include this flag, it defaults to False.",
    )
    subparser.add_argument(
        "--do_not_balance_hdf5",
        action="store_true",
        help="If you DO NOT want to balance hdf5 files, this is not recommended unless the you are dealing with a \
        huge amount of data (many terabytes), or do not want shuffling among splits.",  # noqa: E251
    )
    subparser.add_argument(
        "--num_workers",
        default=min(cpu_count(), 16),
        type=int,
        required=False,
        help="The number of CPU workers to multi-process run tokenization over, if the previous run failed you need to \
        decrease this number.",  # noqa: E251
    )
    subparser.add_argument(
        "--keep_split_jsonls",
        action="store_true",
        help="If you DO NOT want to delete split jsonls files that are in text format, include this flag. \
        The only reason you would include this flag is if you want to see what text is in what hdf5. \
        Including this flag will over 2x the storage space taken up by your dataset.",  # noqa: E251
    )

    # add arguments that are required to be passed on to generative_data_prep/data_prep/data_prep.py
    data_prep_arg_builder(subparser)


def get_args() -> argparse.Namespace:
    """Get the command line arguments.

    Returns:
        The command line arguments as a argparse Namespace.
    """
    parser = argparse.ArgumentParser(description="Text Processing Pipeline")
    subparsers = parser.add_subparsers(dest="cmd")
    # create the train test data subparser
    pipeline_subparser = subparsers.add_parser("pipeline")
    add_pipeline_args(pipeline_subparser)
    # create the create data subparser
    data_prep_subparser = subparsers.add_parser("data_prep")
    add_data_prep_args(data_prep_subparser)
    return parser.parse_args()


def add_special_tokens_dict(tokenizer: PreTrainedTokenizerBase, special_tokens_dict: str):
    """Add the special tokens dictionary to tokenizer.

    Args:
        tokenizer: tokenizer to add special tokens to
        special_tokens_dict: special tokens dictionary
    """
    logger.info(SEP_STR)
    logger.info("Adding special tokens dict:")
    logger.info(special_tokens_dict)
    dict_string = special_tokens_dict.replace("'", '"')
    tokenizer.add_special_tokens(json.loads(dict_string))


def get_tokenizer(
    pretrained_tokenizer: Optional[str],
    tokenizer_class: Optional[str],
    vocab_file: str,
    merges_file: str,
    special_tokens_dict: Optional[str],
) -> PreTrainedTokenizerBase:
    """Create a tokenizer based on input arguments.

    Args:
        pretrained_tokenizer: key to load pretrained tokenizer from huggingface using AutoTokenizer.from_pretrained
        tokenizer_class: class of tokenizer, must be from TOKENIZER_CLASSES
        vocab_file: path to vocab file
        merges_file: path to merges file
        special_tokens_dict: string representation of special tokens dictionary

    Raises:
        ValueError: If the input arguments are not compatible
        NotImplementedError: If the tokenizer class selected has not been implemented

    Returns:
        Tokenizer
    """
    if pretrained_tokenizer is None and tokenizer_class is None:
        pretrained_tokenizer = GPT2_KEY

    if not pretrained_tokenizer and not (merges_file and vocab_file and tokenizer_class):
        err_msg = "You must include either --pretrained_tokenizer, \
        or all three flags: --merges_file, --vocab_file and --tokenizer_class"

        raise ValueError(err_msg)

    if pretrained_tokenizer and (merges_file or vocab_file or tokenizer_class):
        err_msg = "You may not include --pretrained_tokenizer along with any of the following flags: \
        --merges_file, --vocab_file and --tokenizer_class"

        raise ValueError(err_msg)

    if pretrained_tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    else:
        verify_input_file(vocab_file)
        verify_input_file(merges_file)
        for tokenizer_key_i, tokenizer_class_i in TOKENIZER_CLASSES.items():
            if tokenizer_class == tokenizer_key_i:
                tokenizer = tokenizer_class_i(vocab_file, merges_file)

        if tokenizer is None:
            raise NotImplementedError(
                f"The tokenizer_class you selected ({args.tokenizer_class}) has not been implemented "
            )

    if special_tokens_dict:
        add_special_tokens_dict(tokenizer, special_tokens_dict)

    return tokenizer


def get_output_dir(cmd, output_path, overwrite_output_path):
    """Get the output directory based in the input arguments, if input path then return parent dir.

    Args:
        cmd: whether to run 'pipeline' or 'data_prep'
        output_path: If running 'data_prep', what hdf5 file to output into. If running pipeline,
            what directory to output to
        overwrite_output_path: If there is permission ot overwrite output path.

    Raises:
        ValueError: If the input paths are invalid

    Returns:
        Output directory path string
    """
    if cmd == "pipeline":
        verify_output_dir(output_path, overwrite_output_path)
        output_dir = output_path
    elif args.cmd == "data_prep":
        verify_output_file(output_path, overwrite_output_path)
        if os.path.splitext(output_path)[-1] != ".hdf5":
            raise ValueError(f"The output path {output_path} does not end with .hdf5")
        output_dir = os.path.dirname(output_path)

    return output_dir


def get_categories(categories_path: str):
    """Returns a dictionary mapping each category to the corresponding ID.

    Args:
        categories_path: The path to a json file
    """
    category_to_id = None
    if categories_path is not None:
        category_to_id = {}
        if os.path.exists(categories_path):
            _, file_extension = os.path.splitext(categories_path)
            if file_extension != ".json":
                raise ValueError(f"Your --categories_path flag must point to a json file, you used {categories_path}")
            with open(categories_path, "r") as categories_file:
                categories_list = json.load(categories_file)
                if not isinstance(categories_list, list):
                    err_msg = (
                        "Your --categories_path flag must point to a json file that contains a list of categories,"
                    )
                    err_msg += f"the loaded json file instead contains {categories_list}"
                    raise ValueError(err_msg)
        else:
            raise ValueError("Invalid category file path {}, does not exist")

        for id, category in enumerate(categories_list):
            category_to_id[category] = id

    return category_to_id


def create_logger(log_file_path: str, output_dir: str):
    """If log_file_path is defined then return it, otherwise return output_dir/logs.log.

    Args:
        log_file_path: The input log_file_path flag.
        output_dir: The output directory to default to if log_file_path is None.
    """
    if log_file_path is None:
        log_file_path = os.path.join(output_dir, "logs.log")

    # Create a custom log formatter
    formatter = logging.Formatter(" %(message)s")

    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)


if __name__ == "__main__":
    args = get_args()
    if os.path.splitext(args.input_file_path)[1] not in FileExtension.as_list():
        err_msg = f"The input file is not a jsonl or txt file {args.input_file_path}"
        raise ValueError(err_msg)
    verify_input_file(args.input_file_path)
    output_dir = get_output_dir(args.cmd, args.output_path, args.overwrite_output_path)
    create_logger(args.log_file_path, output_dir)

    tokenizer = get_tokenizer(
        args.pretrained_tokenizer,
        args.tokenizer_class,
        args.vocab_file,
        args.merges_file,
        args.special_tokens_dict,
    )
    category_to_id = get_categories(args.categories_path)

    if args.cmd == "pipeline":
        metrics = pipeline_main(
            args.input_file_path,
            tokenizer,
            output_dir,
            args.disable_space_separator,
            args.keep_prompt_only_sequences,
            args.prompt_keyword,
            args.completion_keyword,
            args.shuffle,
            args.overwrite_output_path,
            args.num_workers,
            args.do_not_balance_hdf5,
            args.keep_split_jsonls,
            args.max_seq_length,
            args.input_packing_config,
            args.packing_boundary,
            args.attention_boundary,
            args.num_training_splits,
            args.num_dev_splits,
            args.num_test_splits,
            args.dev_ratio,
            args.test_ratio,
            category_to_id,
            args.prompt_prefix,
            args.prompt_postfix,
        )
    elif args.cmd == "data_prep":
        metrics = data_prep_main(
            args.silent,
            tokenizer,
            args.input_file_path,
            args.output_path,
            args.max_seq_length,
            args.input_packing_config,
            args.packing_boundary,
            args.attention_boundary,
            args.disable_space_separator,
            args.keep_prompt_only_sequences,
            args.prompt_keyword,
            args.completion_keyword,
            category_to_id,
            args.prompt_prefix,
            args.prompt_postfix,
        )

    logger.info("\n")
    logger.info(SEP_STR)
    logger.info("---------METRICS---------")
    logger.info(SEP_STR)
    logger.info(metrics)
    logger.info("\n")
    logger.info(SEP_STR)
    logger.info("---------COMPLETE---------")
