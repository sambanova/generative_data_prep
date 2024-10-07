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
"""

import os
import tempfile
from typing import Dict

import pytest
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    PreTrainedTokenizerBase,
)

from generative_data_prep.__main__ import run_with_training_args
from generative_data_prep.data_prep import data_prep_main, pipeline_main
from generative_data_prep.utils import BoundaryType, PackingConfig
from tests.conftest import TESTS_EXAMPLES_PATH

from .test_utils import (
    check_balance,
    check_diff_hdf5,
    check_no_split_dir,
    check_pipeline,
    check_splits,
)

TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
MODEL_CONFIG = GPT2Config.from_pretrained("gpt2")
LLAMA_TOKENIZER = AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite")


def get_input_path(test_name: str) -> str:
    """Create an absolute path to example input."""
    base_path = TESTS_EXAMPLES_PATH / test_name / f"example_{test_name}_data"
    if os.path.isdir(base_path):
        return base_path
    else:
        ext = ".txt" if "txt" in test_name else ".jsonl"
        return f"{base_path}{ext}"


def gold_data_prep_path(test_name: str) -> str:
    """Create a absolute path to example gold file."""
    return TESTS_EXAMPLES_PATH / test_name / f"data_prepped_{test_name}.hdf5"


def gold_pipeline_path(test_name: str) -> str:
    """Create a absolute path to example gold file."""
    return TESTS_EXAMPLES_PATH / test_name / f"pipelined_{test_name}"


@pytest.mark.parametrize(
    "test_name,tokenizer,max_seq_length,input_packing_config,packing_boundary,attention_boundary,apply_chat_template",
    [
        ("data_prep_test", TOKENIZER, 1024, PackingConfig.get_default(), BoundaryType.JSONL, BoundaryType.JSONL, False),
        (
            "pretraining_txt",
            TOKENIZER,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            False,
        ),
        ("pretraining", TOKENIZER, 1024, PackingConfig.get_default(), BoundaryType.JSONL, BoundaryType.JSONL, False),
        (
            "generative_tuning",
            TOKENIZER,
            1024,
            PackingConfig.from_str("single::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            False,
        ),
        (
            "dialogue",
            TOKENIZER,
            1024,
            PackingConfig.from_str("single::truncate_right"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            False,
        ),
        (
            "metaICL",
            TOKENIZER,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.PROMPT_COMPLETION_PAIR,
            BoundaryType.JSONL,
            False,
        ),
        (
            "apply_chat_template",
            LLAMA_TOKENIZER,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            True,
        ),
    ],
)
def test_data_prep(
    tokenizer: PreTrainedTokenizerBase,
    test_name: str,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    apply_chat_template: bool,
):
    """Test the data prep function."""
    input_path = get_input_path(test_name)
    gold_path = gold_data_prep_path(test_name)
    with tempfile.TemporaryDirectory() as output_dir:
        output_file = os.path.join(output_dir, "test_out.hdf5")
        data_prep_main(
            silent=True,
            tokenizer=tokenizer,
            input_file=input_path,
            output_file=output_file,
            error_log_dir="",
            max_seq_length=max_seq_length,
            input_packing_config=input_packing_config,
            packing_boundary=packing_boundary,
            attention_boundary=attention_boundary,
            disable_space_separator=False,
            keep_prompt_only_sequences=True,
            ignore_input_format_error=False,
            prompt_keyword="prompt",
            completion_keyword="completion",
            apply_chat_template=apply_chat_template,
        )
        check_diff_hdf5(output_file, gold_path)


"""
('pipeline_test', False, 'prompt', 'completion', 'False', True, 1024,
      PackingConfig.get_default(), BoundaryType.JSONL, BoundaryType.JSONL,
      None, None, None, 0.2, 0.1),
('pretraining', False, 'prompt', 'completion', 'False', False, 1024,
      PackingConfig.get_default(), BoundaryType.JSONL, BoundaryType.JSONL,
      32, 0, 0, None, None),
('generative_tuning', False, 'prompt', 'completion', 'False', False, 1024,
      PackingConfig.from_str('single::drop'), BoundaryType.JSONL, BoundaryType.JSONL,
      32, 0, 0, None, None),
('dialogue', False, 'prompt', 'completion', 'False', False, 1024,
      PackingConfig.from_str('single::truncate_right'), BoundaryType.JSONL,
      BoundaryType.JSONL, 32, 0, 0, None, None),
('metaICL', False, 'prompt', 'completion', 'False', False, 1024,
     PackingConfig.from_str('greedy::drop'),
      BoundaryType.PROMPT_COMPLETION_PAIR, BoundaryType.JSONL, 32, 0, 0, None, None)
"""


@pytest.mark.parametrize(
    "test_name,disable_space_separator,keep_prompt_only_sequences,ignore_input_format_error,\
    prompt_keyword,completion_keyword,shuffle,do_not_balance_hdf5,keep_split_jsonls,max_seq_length,\
    input_packing_config,packing_boundary,attention_boundary,num_training_splits,num_dev_splits,\
    num_test_splits,category_to_id,dev_ratio,test_ratio,tokenizer,apply_chat_template",
    [
        (
            "pipeline_test",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            True,
            True,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            None,
            None,
            None,
            None,
            0.2,
            0.1,
            TOKENIZER,
            False,
        ),
        (
            "pretraining_txt",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "pretraining",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "generative_tuning",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("single::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "no_split_dir",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("single::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "dialogue",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("single::truncate_right"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "metaICL",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.PROMPT_COMPLETION_PAIR,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "category_ids",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            2048,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.PROMPT_COMPLETION_PAIR,
            BoundaryType.JSONL,
            32,
            0,
            0,
            ["example_category_1", "example_category_2", "example_category_3"],
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "apply_chat_template",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            16,
            0,
            0,
            None,
            None,
            None,
            LLAMA_TOKENIZER,
            True,
        ),
        (
            "json_load_error_test",
            False,
            True,
            True,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
        (
            "directory_input",
            False,
            True,
            True,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            32,
            0,
            0,
            None,
            None,
            None,
            TOKENIZER,
            False,
        ),
    ],
)
def test_pipeline(
    test_name: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    ignore_input_format_error: bool,
    prompt_keyword: str,
    completion_keyword: str,
    shuffle: str,
    do_not_balance_hdf5: bool,
    keep_split_jsonls: bool,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    num_training_splits: int,
    num_dev_splits: int,
    num_test_splits: int,
    category_to_id: Dict[str, int],
    dev_ratio: float,
    test_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    apply_chat_template: bool,
):
    """Test the pipeline function end to end."""
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    input_path = get_input_path(test_name)
    gold_path = gold_pipeline_path(test_name)
    with tempfile.TemporaryDirectory() as output_dir:
        pipeline_main(
            input_path=input_path,
            tokenizer=tokenizer,
            model_config=MODEL_CONFIG,
            output_dir=output_dir,
            disable_space_separator=disable_space_separator,
            keep_prompt_only_sequences=keep_prompt_only_sequences,
            ignore_input_format_error=ignore_input_format_error,
            prompt_keyword=prompt_keyword,
            completion_keyword=completion_keyword,
            shuffle=shuffle,
            overwrite_output_path=False,
            num_workers=num_workers,
            do_not_balance_hdf5=do_not_balance_hdf5,
            keep_split_jsonls=keep_split_jsonls,
            max_seq_length=max_seq_length,
            input_packing_config=input_packing_config,
            packing_boundary=packing_boundary,
            attention_boundary=attention_boundary,
            num_training_splits=num_training_splits,
            num_dev_splits=num_dev_splits,
            num_test_splits=num_test_splits,
            category_to_id=category_to_id,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            apply_chat_template=apply_chat_template,
        )

        check_pipeline(output_dir, gold_path)

        if keep_split_jsonls:
            check_splits(output_dir, gold_path)
        else:
            check_no_split_dir(output_dir, gold_path)

        if not do_not_balance_hdf5:
            check_balance(os.path.join(output_dir))


@pytest.mark.parametrize(
    "test_name, checkpoint_path, number_of_rdus, grad_accum_steps, pef_batch_size,\
        input_packing_config, apply_chat_template, shuffle, max_seq_length",
    [
        (
            "pretraining",
            "openai-community/gpt2",
            1,  # number_of_rdus
            1,  # grad_accum_steps
            1,  # pef_batch_size
            "full",  # input_packing_config
            False,  # apply_chat_template
            "False",
            None,
        ),
        (
            "data_prep_from_main",
            "Qwen/Qwen2-1.5B",
            1,  # number_of_rdus
            1,  # grad_accum_steps
            1,  # pef_batch_size
            "greedy::drop",  # input_packing_config
            True,  # apply_chat_template
            "False",
            4096,
        ),
    ],
)
def test_run_with_training_args(
    test_name: str,
    checkpoint_path: str,
    number_of_rdus: int,
    grad_accum_steps: int,
    pef_batch_size: int,
    input_packing_config: str,
    apply_chat_template: bool,
    shuffle: str,
    max_seq_length: int,
):
    """Test if we can call main function using training arguments."""
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    input_path = get_input_path(test_name)
    gold_path = gold_pipeline_path(test_name)
    with tempfile.TemporaryDirectory() as output_path:
        log_file_path = os.path.join(output_path, "logs.log")
        run_with_training_args(
            input_path,
            output_path,
            log_file_path,
            checkpoint_path,
            number_of_rdus,
            grad_accum_steps,
            pef_batch_size,
            max_seq_length=max_seq_length,
            num_workers=num_workers,
            input_packing_config=input_packing_config,
            apply_chat_template=apply_chat_template,
            shuffle=shuffle,
        )
        check_pipeline(output_path, gold_path)
