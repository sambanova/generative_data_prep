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
import json
import logging
import os
from typing import Optional

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main, pipeline_main
from generative_data_prep.utils import (
    FileExtension,
    add_file_handler,
    check_deprecated_args,
    get_arg_parser,
    get_config_file_path,
    log_current_datetime,
    log_elapsed_time,
    log_git_commit_hash,
    log_input_args,
    log_metrics,
    log_sep_str,
    log_training_details,
    training_to_data_prep_params,
    verify_input_file,
    verify_output_dir,
    verify_output_file,
)

logger = logging.getLogger("generative_data_prep_logger")
logging.config.fileConfig(get_config_file_path())


def add_special_tokens_dict(tokenizer: PreTrainedTokenizerBase, special_tokens_dict: str):
    """Add the special tokens dictionary to tokenizer.

    Args:
        tokenizer: tokenizer to add special tokens to
        special_tokens_dict: special tokens dictionary
    """
    log_sep_str()
    logger.info(f"Adding special tokens dict:\n{special_tokens_dict}")
    dict_string = special_tokens_dict.replace("'", '"')
    tokenizer.add_special_tokens(json.loads(dict_string))


def get_tokenizer(
    pretrained_tokenizer: Optional[str],
    special_tokens_dict: Optional[str],
) -> PreTrainedTokenizerBase:
    """Create a tokenizer based on input arguments.

    Args:
        pretrained_tokenizer: key to load pretrained tokenizer from huggingface using AutoTokenizer.from_pretrained
        special_tokens_dict: string representation of special tokens dictionary

    Raises:
        ValueError: If the input arguments are not compatible
        NotImplementedError: If the tokenizer class selected has not been implemented

    Returns:
        Tokenizer, ModelConfig
    """
    tokenizer = None
    model_config = None

    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    model_config = AutoConfig.from_pretrained(pretrained_tokenizer)

    if special_tokens_dict:
        add_special_tokens_dict(tokenizer, special_tokens_dict)

    return tokenizer, model_config


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
    elif cmd == "data_prep":
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


def main(args):
    """Wrapping function instead of putting into __main__."""
    output_dir = get_output_dir(args.cmd, args.output_path, args.overwrite_output_path)
    add_file_handler(args.log_file_path, output_dir)
    log_git_commit_hash()
    log_current_datetime()
    log_input_args(args)

    json_error_log_dir = os.path.join(output_dir, "json_error_log")
    verify_output_dir(json_error_log_dir, True)

    tokenizer, model_config = get_tokenizer(
        args.pretrained_tokenizer,
        args.special_tokens_dict,
    )
    category_to_id = get_categories(args.categories_path)

    if not os.path.isdir(args.input_path):
        if os.path.splitext(args.input_path)[1] not in FileExtension.as_list():
            err_msg = f"The input file is not a jsonl or txt file {args.input_path}"
            raise ValueError(err_msg)
        verify_input_file(args.input_path)

    if args.cmd == "pipeline":
        metrics, dataset_metadata = pipeline_main(
            args.input_path,
            tokenizer,
            model_config,
            output_dir,
            args.disable_space_separator,
            args.keep_prompt_only_sequences,
            args.ignore_input_format_error,
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
            args.apply_chat_template,
        )
    elif args.cmd == "data_prep":
        metrics = data_prep_main(
            args.silent,
            tokenizer,
            args.input_path,
            args.output_path,
            json_error_log_dir,
            args.max_seq_length,
            args.input_packing_config,
            args.packing_boundary,
            args.attention_boundary,
            args.disable_space_separator,
            args.keep_prompt_only_sequences,
            args.ignore_input_format_error,
            args.prompt_keyword,
            args.completion_keyword,
            category_to_id,
            args.prompt_prefix,
            args.prompt_postfix,
            apply_chat_template=args.apply_chat_template,
        )

    log_metrics(metrics)
    log_elapsed_time()
    if args.cmd == "pipeline":
        log_training_details(dataset_metadata)

    return metrics


def run_with_training_args(
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
):
    """Runs the main pipeline for data preparation and training configuration based on provided arguments.

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
        custom_tokenizer_path (str, optional): Path to a custom tokenizer, if any. If not provided, defaults to
            the tokenizer from `checkpoint_path`.
        input_packing_config (str, optional): Strategy for packing input sequences.
            Defaults to "greedy::drop".
        apply_chat_template (bool, optional): Whether to apply chat formatting to inputs. If None,
            it is inferred based on the tokenizer's capabilities. Defaults to None.

    Returns:
       metrics: Dataset metrics related to the dataset we just prepared.

    Raises:
        ValueError: If the arguments provided to `training_to_data_prep_params` are invalid,
            such as incompatible tokenizers or missing required parameters.
    """
    data_prep_args = training_to_data_prep_params(
        input_path,
        output_path,
        log_file_path,
        checkpoint_path,
        number_of_rdus,
        grad_accum_steps,
        pef_batch_size,
        max_seq_length,
        evaluation_ratio,
        num_workers,
        custom_tokenizer_path,
        input_packing_config,
        apply_chat_template,
        shuffle,
    )
    data_prep_args = check_deprecated_args(data_prep_args)

    return main(data_prep_args)


if __name__ == "__main__":
    parser = get_arg_parser()
    data_prep_args = parser.parse_args()
    data_prep_args = check_deprecated_args(data_prep_args)
    main(data_prep_args)
