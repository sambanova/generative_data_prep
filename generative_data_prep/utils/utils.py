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

import argparse
import hashlib
import json
import math
import os
import shutil
from subprocess import PIPE, run  # nosec
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from .arg_configs import PackingConfig
from .constants import BoundaryType
from .logger import LOGGER, log_sep_str


class TokenizerConfigPair:
    """New datastructure to store tokenizer and config pairs for same model type."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: PretrainedConfig) -> None:
        """Initialization of TokenizerConfigPair.

        Args:
            tokenizer: Tokenizer associated with key
            config: Config associated with key
        """
        self.tokenizer = tokenizer
        self.config = config


GPT2_KEY = "gpt2"
TOKENIZER_CLASSES = {GPT2_KEY: TokenizerConfigPair(tokenizer=GPT2Tokenizer, config=GPT2Config)}


def data_prep_arg_builder(parser: argparse.ArgumentParser):
    """Adds all the arguments that are required for data_prep.py's argparser, besides the output_path.

    Args:
        parser (argparse.ArgumentParser): parser to add arguments to
    """
    parser.add_argument("--input_file_path", type=str, required=True, help="The input jsonl file path.")
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
            return False
        current_modified_time = os.path.getmtime(file)
        current_size = os.path.getsize(file)
        if current_size != file_info_dict[hash_file_name]["size"] or not math.isclose(
            current_modified_time, file_info_dict[hash_file_name]["modified_time"]
        ):
            file_hash = file_info_dict[hash_file_name]["sha256"]
            current_file_hash = _calculate_sha256(file)
            if file_hash != current_file_hash:
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


def add_special_tokens_dict(tokenizer: PreTrainedTokenizerBase, special_tokens_dict: str):
    """Add the special tokens dictionary to tokenizer.

    Args:
        tokenizer: tokenizer to add special tokens to
        special_tokens_dict: special tokens dictionary
    """
    log_sep_str()
    LOGGER.info(f"Adding special tokens dict:\n{special_tokens_dict}")
    dict_string = special_tokens_dict.replace("'", '"')
    tokenizer.add_special_tokens(json.loads(dict_string))


def get_tokenizer(
    pretrained_tokenizer: Optional[str],
    tokenizer_class: Optional[str],
    vocab_file: Optional[str],
    merges_file: Optional[str],
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
        Tokenizer, ModelConfig
    """
    tokenizer = None
    model_config = None
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
        model_config = AutoConfig.from_pretrained(pretrained_tokenizer)
    else:
        if tokenizer_class in TOKENIZER_CLASSES:
            tokenizer = TOKENIZER_CLASSES[tokenizer_class].tokenizer(vocab_file, merges_file)
            model_config = TOKENIZER_CLASSES[tokenizer_class].config(vocab_size=tokenizer.vocab_size)

        if tokenizer is None:
            raise NotImplementedError(f"The tokenizer_class you selected ({tokenizer_class}) has not been implemented")
        if model_config is None:
            raise NotImplementedError(
                f"The tokenizer_class you selected ({tokenizer_class}) is missing a config associated with it"
            )

    if special_tokens_dict:
        add_special_tokens_dict(tokenizer, special_tokens_dict)

    return tokenizer, model_config
