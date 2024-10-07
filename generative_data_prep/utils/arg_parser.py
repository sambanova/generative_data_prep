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


This module implements argument parsing utilities for a data preparation pipeline.

It provides functions to set up command-line arguments for two main scripts:
1. data_prep.py: For processing and tokenizing input data.
2. pipeline.py: For running the entire data preparation pipeline, including splitting data into train/dev/test sets.

The module defines three main functions:
- data_prep_arg_builder: Adds arguments required for data_prep.py
- add_data_prep_args: Creates the argument parser for data_prep.py
- add_pipeline_args: Creates the argument parser for pipeline.py
- get_arg_parser: Returns the complete argument parser for the entire pipeline
"""
import argparse
from multiprocessing import cpu_count

from .arg_configs import PackingConfig
from .constants import TOKENIZER_CLASSES, BoundaryType


def data_prep_arg_builder(parser: argparse.ArgumentParser):
    """Adds all the arguments that are required for data_prep.py's argparser, besides the output_path.

    Args:
        parser (argparse.ArgumentParser): parser to add arguments to
    """
    parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        help="The input jsonl file path or a path to a directory of input jsonls.",
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        required=False,
        help="Deprecated argument",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "The path to the output directory if using end to end data preparation, the path to output hdf5 file if"
            " running tokenization"
        ),
    )
    parser.add_argument(
        "--log_file_path",
        type=str,
        required=False,
        default=None,
        help="The log file path. Defaults to '<output_path>/logs.log'",
    )
    parser.add_argument(
        "--overwrite_output_path",
        action="store_true",
        help="If the file or files stored at the output path can be over-written",
    )

    parser.add_argument(
        "--ignore_input_format_error",
        action="store_true",
        help=(
            "Ignore format errors in the input jsonl file by skipping misformatted lines."
            " Warning: these lines are dropped from the generated dataset"
        ),
    )

    parser.add_argument(
        "--tokenizer_class",
        type=str,
        choices=list(TOKENIZER_CLASSES.keys()),
        default=None,
        required=False,
        help=(
            "pre-specified tokenizer class to run, defaults to gpt2, must be a choice from"
            f" {list(TOKENIZER_CLASSES.keys())}"
        ),
    )
    parser.add_argument(
        "--pretrained_tokenizer",
        default=None,
        type=str,
        required=False,
        help=(
            "The pretrained tokenizer to be used, loaded using"
            " transformers.AutoTokenizer.from_pretrained(args.pretrained_tokenizer), in lieu of a custom vocab and"
            " merges file."
        ),
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        required=False,
        help=(
            "The vocabulary file for the tokenizer. Should be a .json file for the tokenizer class specified by"
            " --tokenizer_class."
        ),
    )
    parser.add_argument(
        "--merges_file",
        type=str,
        default=None,
        required=False,
        help="The merges file to be used with the tokenizer class specified by --tokenizer_class.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=2048,
        type=int,
        required=False,
        help=(
            "The max sequence length after tokenization. \n Sequence will be truncated or padded to this length before"
            " input into the model. Defaults to 512."
        ),
    )
    parser.add_argument(
        "--input_packing_config",
        type=PackingConfig.from_str,
        default=PackingConfig.get_default(),
        choices=PackingConfig.get_choices(),
        required=False,
        help=(
            "The first argument in the packing config defines the method of placing text into sequences, the second"
            " argument defines how to handle jsonls that do not fit within the max_seq_length. 'full': Defines the"
            " entire packing config, Completely fill sequences with tokens, as soon as sequences is full start packing"
            " into new sequence. Ignore article boundaries, they may be split across multiple sequences. 'greedy': Fit"
            " as many articles as possible into a sequence, make sure no article is split across multiple sequences."
            " Fill the left over space in each sequence with padding. 'single': Each sequence contains only 1 article. "
            " Fill the rest of the sequence with padding.  'drop': Drop the entire article if there are any tokens that"
            " overflow beyond the max sequence length.  'truncate_left':  Truncate the article from the left if there"
            " are any tokens that overflow beyond the max sequence length.  'truncate_right':  Truncate the article"
            " from the right if there are any tokens that overflow beyond the max sequence length."
        ),
    )
    parser.add_argument(
        "--packing_boundary",
        type=str,
        default=BoundaryType.JSONL.value,
        choices=BoundaryType.as_list(),
        required=False,
        help=(
            "How to define the boundary when packing jsonl into sequences. Choosing jsonl will define each jsonl as a"
            " packing unit, and keep it together. Choosing prompt_completion_pair option, defines"
            " prompt_completion_pairs as the packing unit and will keep them together, but prompt completion pairs"
            " within one jsonl may be split into multiple sequences."
        ),
    )
    parser.add_argument(
        "--attention_boundary",
        type=str,
        default=BoundaryType.JSONL.value,
        choices=BoundaryType.as_list(),
        required=False,
        help=(
            "What boundary to use when training with --article_attention flag. If you choose prompt_completion_pair"
            " tokens will only attend to tokens in the prompt_completion_pair. If you choose jsonl, then tokens will"
            " attend to all the prompt completion pairs in the jsonl"
        ),
    )
    parser.add_argument(
        "--special_tokens_dict",
        type=str,
        default=None,
        required=False,
        help="Any non-standard special tokens in JSON format to add to tokenizer. e.g. '{'sep_token': \"[SEP]\"}'",
    )
    parser.add_argument(
        "--prompt_keyword",
        default="prompt",
        type=str,
        required=False,
        help="keyword used in input json to specify prompt",
    )
    parser.add_argument(
        "--completion_keyword",
        default="completion",
        type=str,
        required=False,
        help="keyword used in input json to specify completion, defaults to 'completion",
    )

    parser.add_argument(
        "--prompt_prefix",
        default=None,
        type=str,
        required=False,
        help="Text to add before the prompt, for chatML conventions use",
    )
    parser.add_argument(
        "--prompt_postfix",
        default=None,
        type=str,
        required=False,
        help="text to add after the prompt, for chatML conventions use",
    )
    parser.add_argument(
        "--disable_space_separator",
        action="store_true",
        help=(
            "FOR ADVANCED USERS: If you include this flag, NO spaces will be appended to the completion. (If you do not"
            " add this flag then a space is added to every completion if it does not already have a space) This flag is"
            ' dangerous because if you have input data like {"prompt": hello. "completion": how are you?}, when the'
            ' prompt and completion are combined it will look like "hello.how are you?" which will mess up the'
            " tokenization."
        ),
    )
    parser.add_argument(
        "--keep_prompt_only_sequences",
        action="store_true",
        help=(
            "FOR ADVANCED USERS: If you include this flag, packed sequences with only prompt tokens will not be"
            " dropped. Data with only prompt will be dropped by default because training with prompt-only sequences"
            " with prompt_loss_weight=0.0 may lead to errors. Data is dropped because of one of the following"
            " conditions: 1. the input file data prompt completion pairs contains only a prompt. 2. If the sequence is"
            " truncated such that only prompt tokens remain"
        ),
    )
    parser.add_argument(
        "--categories_path",
        default=None,
        type=str,
        required=False,
        help=(
            "If you include this flag, then the 'category' field from your input jsonls will be stored in the"
            " 'category_id' dataset in your output hdf5 files. This flag must point to the file path of a json"
            " file that contains a list of all the strings of the 'category' keys in your dataset."
        ),
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="If you want to apply the chat template to your data, include this flag. \
        The chat template is a template that is applied to the data to make it more conversational \
        which includes a role, like “user” or “assistant”, as well as message text. \
        This chat template will be extracted from tokenizer.apply_chat_template.",
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


def get_arg_parser() -> argparse.ArgumentParser:
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

    return parser
