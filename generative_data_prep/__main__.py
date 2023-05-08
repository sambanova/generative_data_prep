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


Entry point to the Text Processing Pipeline.
"""

import argparse
import json
import os
from multiprocessing import cpu_count
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main, pipeline_main
from generative_data_prep.utils import (GPT2_KEY, SEP_STR, TOKENIZER_CLASSES,
                                        FileExtension, data_prep_arg_builder,
                                        verify_input_file, verify_output_dir,
                                        verify_output_file)


def add_data_prep_args(subparser: argparse.ArgumentParser):
    """Create the argparser for the generative_data_prep/data_prep/data_prep.py script.

    Args:
        subparser: The parser to add the arguments to.
    """
    subparser.add_argument('--silent',
                           default=False,
                           action='store_true',
                           required=False,
                           help='Do not allow this script to print')
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
        help=  # noqa: E251
        "The number of training files to split input data into. If you specify the --dev_ratio and --test_ratio \
        flags, The total number of splits will be (num_training_splits / (1-dev_ratio-test_ratio)), and the number \
        of dev and test splits are calculated accordingly. If you specify --num_dev_splits and --num_test_splits \
        flags then those will  directory define the number of splits and therefore the ratios. We recommend you do \
        not include this flag and allow it to default, because the number of training splits must be greater than \
        the number of parallel workers and it is best if the number of training splits is a multiple of the number \
        of workers. It defaults to 32 training splits if input_file_size < 10GB, 128 training splits if 10GB < \
        input_file_size <100GB, 256 training splits if 100GB < input_file_size."
    )
    subparser.add_argument(
        "--dev_ratio",
        default=None,
        type=float,
        required=False,
        help=  # noqa: E251
        "The ratio of data that should be excluded from train set and used for evaluation, defaults to 10%%. If \
        you specify this flag, do not specify --num_dev_splits or --num_test_splits."
    )
    subparser.add_argument(
        "--num_dev_splits",
        default=None,
        type=int,
        required=False,
        help=  # noqa: E251
        "If you do not specify --dev_ratio, you may specify num_dev_splits. If you include this flag, you must \
        also include the --num_dev_splits and --num_training_splits flags")
    subparser.add_argument(
        "--test_ratio",
        default=None,
        type=float,
        required=False,
        help=  # noqa: E251
        "The ratio of data that should be excluded from train set and is saved for testing. This data is not \
        tokenized and left in jsonl format, defaults to 0%%. If you specify this flag, do not specify \
        --num_dev_splits or --num_test_splits.")
    subparser.add_argument(
        "--num_test_splits",
        default=None,
        type=int,
        required=False,
        help=  # noqa: E251
        "If you do not specify --test_ratio, you may specify num_dev_splits. If you include this flag, you must \
        also include the --num_dev_splits and --num_training_splits flags.")
    subparser.add_argument(
        "--shuffle",
        default='False',
        const='False',
        nargs='?',
        choices=['False', 'on_RAM', 'large_file'],
        help=  # noqa: E251
        "Choose the on_RAM option if your file is small enough to fit on RAM (If you are not sure if it fits \
        on RAM, default to this flag). If you are running a linux operating system and your file is too large to fit \
        on RAM, please choose large_file option, this will run approximate file shuffling that can handle files of \
        any size. If you want to do large file shuffling but you are not on linux, please shuffle the file before \
        using this script. If the input file should not be shuffled, do not include this flag, it defaults to False."
    )
    subparser.add_argument(
        "--do_not_balance_hdf5",
        action='store_true',
        help=  # noqa: E251
        "If you DO NOT want to balance hdf5 files, this is not recommended unless the you are dealing with a \
        huge amount of data (many terabytes), or do not want shuffling among splits."
    )
    subparser.add_argument(
        "--num_workers",
        default=cpu_count(),
        type=int,
        required=False,
        help=  # noqa: E251
        "The number of CPU workers to multi-process run tokenization over, if the previous run failed you need to \
        decrease this number.")

    # add arguments that are required to be passed on to generative_data_prep/data_prep/data_prep.py
    data_prep_arg_builder(subparser)


def get_args() -> argparse.Namespace:
    """Get the command line arguments.

    Returns:
        The command line arguments as a argparse Namespace.
    """
    parser = argparse.ArgumentParser(description='Text Processing Pipeline')
    subparsers = parser.add_subparsers(dest='cmd')
    # create the train test data subparser
    pipeline_subparser = subparsers.add_parser('pipeline')
    add_pipeline_args(pipeline_subparser)
    # create the create data subparser
    data_prep_subparser = subparsers.add_parser('data_prep')
    add_data_prep_args(data_prep_subparser)
    return parser.parse_args()


def add_special_tokens_dict(tokenizer: PreTrainedTokenizerBase,
                            special_tokens_dict: str):
    """Add the special tokens dictionary to tokenizer.

    Args:
        tokenizer: tokenizer to add special tokens to
        special_tokens_dict: special tokens dictionary
    """
    print(SEP_STR)
    print('Adding special tokens dict:')
    print(special_tokens_dict, flush=True)
    dict_string = special_tokens_dict.replace("'", '"')
    tokenizer.add_special_tokens(json.loads(dict_string))


def get_tokenizer(pretrained_tokenizer: Optional[str],
                  tokenizer_class: Optional[str], vocab_file: str,
                  merges_file: str, special_tokens_dict: Optional[str]
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

    if not pretrained_tokenizer and not (merges_file and vocab_file
                                         and tokenizer_class):
        err_msg = 'You must include either --pretrained_tokenizer, \
        or all three flags: --merges_file, --vocab_file and --tokenizer_class'

        raise ValueError(err_msg)

    if (pretrained_tokenizer
            and (merges_file or vocab_file or tokenizer_class)):
        err_msg = 'You may not include --pretrained_tokenizer along with any of the following flags: \
        --merges_file, --vocab_file and --tokenizer_class'

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
                f'The tokenizer_class you selected ({args.tokenizer_class}) has not been implemented '
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
    if cmd == 'pipeline':
        verify_output_dir(output_path, overwrite_output_path)
        output_dir = output_path
    elif args.cmd == 'data_prep':
        verify_output_file(output_path, overwrite_output_path)
        if os.path.splitext(output_path)[-1] != '.hdf5':
            raise ValueError(
                f'The output path {output_path} does not end with .hdf5')
        output_dir = os.path.dirname(output_path)

    return output_dir


if __name__ == '__main__':
    args = get_args()
    err_msg = f'The input file is not a jsonl or txt file {args.input_file_path}'
    assert os.path.splitext(
        args.input_file_path)[1] in FileExtension.as_list(), err_msg
    verify_input_file(args.input_file_path)
    output_dir = get_output_dir(args.cmd, args.output_path,
                                args.overwrite_output_path)
    tokenizer = get_tokenizer(args.pretrained_tokenizer, args.tokenizer_class,
                              args.vocab_file, args.merges_file,
                              args.special_tokens_dict)

    if args.cmd == 'pipeline':
        pipeline_main(args.input_file_path, tokenizer, output_dir,
                      args.disable_space_separator,
                      args.keep_prompt_only_sequences, args.prompt_keyword,
                      args.completion_keyword, args.shuffle,
                      args.overwrite_output_path, args.num_workers,
                      args.do_not_balance_hdf5, args.max_seq_length,
                      args.input_packing_config, args.packing_boundary,
                      args.attention_boundary, args.num_training_splits,
                      args.num_dev_splits, args.num_test_splits,
                      args.dev_ratio, args.test_ratio)
    elif args.cmd == 'data_prep':
        data_prep_main(args.silent, tokenizer, args.input_file_path,
                       args.output_path, args.max_seq_length,
                       args.input_packing_config, args.packing_boundary,
                       args.attention_boundary, args.disable_space_separator,
                       args.keep_prompt_only_sequences, args.prompt_keyword,
                       args.completion_keyword)
