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
from pathlib import Path

import pytest

import generative_data_prep
from generative_data_prep.data_prep import multiprocess_data_prep

from .test_utils import EXAMPLE_PATH


def get_split_dir(test_name: str) -> str:
    """Create a absolute path to example input."""
    return os.path.join(Path.cwd(), EXAMPLE_PATH, test_name, f"pipelined_{test_name}", "splits")


def get_files_to_tokenize(test_name: str) -> str:
    """Create a absolute path to example input."""
    os.listdir(get_split_dir(test_name))


class SomeStrangeException(Exception):
    """A random exception created for the purpose of this test."""

    pass


def data_prep_main_helper_dummy(args):
    """Mock function for the data_prep_main_helper function."""
    raise SomeStrangeException


def test_multiprocess_data_prep_graceful_exit():
    """Tests that the multiprocess_data_prep function exits when a child process raises an Exception."""
    # using unittest.mock.patch wasn't working because of a pickling error when multiprocessing over the Mock object.
    # Apparently there is a library called SharedMock that we can look into, but due to lack of time, just using this
    # hacky import method to mock the data_prep_main_helper function.
    generative_data_prep.data_prep.pipeline.data_prep_main_helper = data_prep_main_helper_dummy
    with pytest.raises(SomeStrangeException):
        multiprocess_data_prep(
            files_to_tokenize=[get_split_dir("generative_tuning")],
            split_dir=get_files_to_tokenize("generative_tuning"),
            hdf5_dir="hdf5",
            max_seq_length=1024,
            input_packing_config=None,
            packing_boundary=None,
            attention_boundary=None,
            prompt_keyword="",
            completion_keyword="",
            disable_space_separator=False,
            keep_prompt_only_sequences=False,
            tokenizer=None,
            num_workers=4,
            input_file_size_in_gb=4,
        )
