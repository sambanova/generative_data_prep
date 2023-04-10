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

TOKENIZER = GPT2Tokenizer.from_pretrained(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 'gpt2_tokenizer'))

INPUT_FILES = 'tests/e2e_test_files'


@pytest.fixture
def input_path(input_name: str) -> str:
    """Create a absolute path to example input."""
    return os.path.join(Path.cwd(), INPUT_FILES, 'input', input_name)


@pytest.fixture
def gold_path(gold_name: str) -> str:
    """Create a absolute path to example gold file."""
    return os.path.join(Path.cwd(), INPUT_FILES, 'gold', gold_name)


@pytest.mark.parametrize(
    'tokenizer,input_name,max_seq_length,input_packing_config,packing_boundary,\
                         attention_boundary,gold_name',
    [(TOKENIZER, 'data_prep_ex.jsonl', 1024, PackingConfig.get_default(),
      BoundaryType.JSONL, BoundaryType.JSONL, 'data_prep_ex.hdf5')])
def test_data_prep(tokenizer: PreTrainedTokenizerBase, input_path: str,
                   max_seq_length: int, input_packing_config: PackingConfig,
                   packing_boundary: BoundaryType,
                   attention_boundary: BoundaryType, gold_path: str):
    with tempfile.TemporaryDirectory() as output_dir:
        output_file = os.path.join(output_dir, 'test_out.hdf5')
        data_prep_main(silent=True,
                       tokenizer=tokenizer,
                       input_file=input_path,
                       output_file=output_file,
                       max_seq_length=max_seq_length,
                       input_packing_config=input_packing_config,
                       packing_boundary=packing_boundary,
                       attention_boundary=attention_boundary,
                       disable_space_separator=False,
                       prompt_keyword='prompt',
                       completion_keyword='completion')
        check_diff_hdf5(output_file, gold_path)


@pytest.mark.parametrize(
    'input_name,gold_name,disable_space_separator,prompt_keyword,completion_keyword,shuffle,do_not_balance_hdf5,\
        max_seq_length,input_packing_config,packing_boundary,attention_boundary,\
            num_training_splits,num_dev_splits,num_test_splits,dev_ratio,test_ratio',
    [('pipeline_ex_data.jsonl', 'pipeline_no_balanced', False, 'prompt',
      'completion', 'on_RAM', True, 1024, PackingConfig.get_default(),
      BoundaryType.JSONL, BoundaryType.JSONL, None, None, None, 0.2, 0.1),
     ('pipeline_ex_data.jsonl', 'pipeline_balanced', False, 'prompt',
      'completion', 'on_RAM', False, 1024, PackingConfig.get_default(),
      BoundaryType.JSONL, BoundaryType.JSONL, 20, 5, 0, None, None)])
def test_pipeline(input_path: str, gold_path: str,
                  disable_space_separator: bool, prompt_keyword: str,
                  completion_keyword: str, shuffle: str,
                  do_not_balance_hdf5: bool, max_seq_length: int,
                  input_packing_config: PackingConfig,
                  packing_boundary: BoundaryType,
                  attention_boundary: BoundaryType, num_training_splits: int,
                  num_dev_splits: int, num_test_splits: int, dev_ratio: float,
                  test_ratio: float):
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    with tempfile.TemporaryDirectory() as output_dir:
        pipeline_main(input_file_path=input_path,
                      tokenizer=TOKENIZER,
                      output_dir=output_dir,
                      disable_space_separator=disable_space_separator,
                      prompt_keyword=prompt_keyword,
                      completion_keyword=completion_keyword,
                      shuffle=shuffle,
                      overwrite_output_path=False,
                      num_workers=num_workers,
                      do_not_balance_hdf5=do_not_balance_hdf5,
                      max_seq_length=max_seq_length,
                      input_packing_config=input_packing_config,
                      packing_boundary=packing_boundary,
                      attention_boundary=attention_boundary,
                      num_training_splits=num_training_splits,
                      num_dev_splits=num_dev_splits,
                      num_test_splits=num_test_splits,
                      dev_ratio=dev_ratio,
                      test_ratio=test_ratio)

        check_pipeline(output_dir, gold_path)

        if not do_not_balance_hdf5:
            check_balance(os.path.join(output_dir, 'hdf5'))
