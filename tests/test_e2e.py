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
from pathlib import Path

import pytest
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase

from generative_data_prep.data_prep import data_prep_main, pipeline_main
from generative_data_prep.utils import BoundaryType, PackingConfig

from .test_utils import check_balance, check_diff_hdf5, check_pipeline

TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

EXAMPLE_PATH = "tests/examples"


def get_input_path(test_name: str) -> str:
    """Create a absolute path to example input."""
    ext = ".txt" if "txt" in test_name else ".jsonl"
    return os.path.join(Path.cwd(), EXAMPLE_PATH, test_name, f"example_{test_name}_data{ext}")


def gold_data_prep_path(test_name: str) -> str:
    """Create a absolute path to example gold file."""
    return os.path.join(Path.cwd(), EXAMPLE_PATH, test_name, f"data_prepped_{test_name}.hdf5")


def gold_pipeline_path(test_name: str) -> str:
    """Create a absolute path to example gold file."""
    return os.path.join(Path.cwd(), EXAMPLE_PATH, test_name, f"pipelined_{test_name}")


@pytest.mark.parametrize(
    "test_name,tokenizer,max_seq_length,input_packing_config,packing_boundary,attention_boundary",
    [
        (
            "data_prep_test",
            TOKENIZER,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
        ),
        (
            "pretraining_txt",
            TOKENIZER,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
        ),
        (
            "pretraining",
            TOKENIZER,
            1024,
            PackingConfig.get_default(),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
        ),
        (
            "generative_tuning",
            TOKENIZER,
            1024,
            PackingConfig.from_str("single::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
        ),
        (
            "dialogue",
            TOKENIZER,
            1024,
            PackingConfig.from_str("single::truncate_right"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
        ),
        (
            "metaICL",
            TOKENIZER,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.PROMPT_COMPLETION_PAIR,
            BoundaryType.JSONL,
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
            max_seq_length=max_seq_length,
            input_packing_config=input_packing_config,
            packing_boundary=packing_boundary,
            attention_boundary=attention_boundary,
            disable_space_separator=False,
            keep_prompt_only_sequences=True,
            prompt_keyword="prompt",
            completion_keyword="completion",
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
    "test_name,disable_space_separator,keep_prompt_only_sequences,prompt_keyword,completion_keyword,\
    shuffle,do_not_balance_hdf5,do_not_delete_split_jsonls,max_seq_length,input_packing_config,packing_boundary,\
    attention_boundary,num_training_splits,num_dev_splits,num_test_splits,dev_ratio,test_ratio",
    [
        (
            "pipeline_test",
            False,
            True,
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
            0.2,
            0.1,
        ),
        (
            "pretraining_txt",
            False,
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
        ),
        (
            "pretraining",
            False,
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
        ),
        (
            "generative_tuning",
            False,
            True,
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
        ),
        (
            "dialogue",
            False,
            True,
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
        ),
        (
            "metaICL",
            False,
            True,
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
        ),
    ],
)
def test_pipeline(
    test_name: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    prompt_keyword: str,
    completion_keyword: str,
    shuffle: str,
    do_not_balance_hdf5: bool,
    do_not_delete_split_jsonls: bool,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    num_training_splits: int,
    num_dev_splits: int,
    num_test_splits: int,
    dev_ratio: float,
    test_ratio: float,
):
    """Test the pipeline function end to end."""
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    input_path = get_input_path(test_name)
    gold_path = gold_pipeline_path(test_name)
    with tempfile.TemporaryDirectory() as output_dir:
        pipeline_main(
            input_file_path=input_path,
            tokenizer=TOKENIZER,
            output_dir=output_dir,
            disable_space_separator=disable_space_separator,
            keep_prompt_only_sequences=keep_prompt_only_sequences,
            prompt_keyword=prompt_keyword,
            completion_keyword=completion_keyword,
            shuffle=shuffle,
            overwrite_output_path=False,
            num_workers=num_workers,
            do_not_balance_hdf5=do_not_balance_hdf5,
            do_not_delete_split_jsonls=do_not_delete_split_jsonls,
            max_seq_length=max_seq_length,
            input_packing_config=input_packing_config,
            packing_boundary=packing_boundary,
            attention_boundary=attention_boundary,
            num_training_splits=num_training_splits,
            num_dev_splits=num_dev_splits,
            num_test_splits=num_test_splits,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
        )
        check_pipeline(output_dir, gold_path)

        if not do_not_balance_hdf5:
            check_balance(os.path.join(output_dir, "hdf5"))
