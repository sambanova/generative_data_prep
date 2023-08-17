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

import h5py
import numpy as np
import pytest

from generative_data_prep.data_buffers import Hdf5FileBuffer
from generative_data_prep.tokenized_line import TokenizedSequence

MAX_SEQ_LENGTH = 4
DATA_TYPE = "i4"
CHUNK_SIZE_1 = np.dtype(DATA_TYPE).itemsize * MAX_SEQ_LENGTH
CHUNK_SIZE_2 = CHUNK_SIZE_1 * 2
CHUNK_SIZE_3 = CHUNK_SIZE_1 * 3
EOS = 9


@pytest.mark.fast
@pytest.mark.parametrize(
    "max_seq_length1,length,max_seq_length,eos_token_id,chunk_size,num_iterations,gold_input_ids,gold_token_type_ids",
    [
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_1,
            1,
            np.array([[0, 1, 2, 3]]),
            np.array([[0, -1, -2, -3]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_2,
            1,
            np.array([[0, 1, 2, 3]]),
            np.array([[0, -1, -2, -3]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_3,
            1,
            np.array([[0, 1, 2, 3]]),
            np.array([[0, -1, -2, -3]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_1,
            2,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_2,
            2,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_3,
            2,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_1,
            3,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_2,
            3,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]]),
        ),
        (
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            MAX_SEQ_LENGTH,
            EOS,
            CHUNK_SIZE_3,
            3,
            np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
            np.array([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]]),
        ),
    ],
)
def test_hfd5_text_buffer_write(
    max_seq_length1: int,
    tokenized_line: TokenizedSequence,
    chunk_size: int,
    num_iterations: int,
    gold_input_ids: np.ndarray,
    gold_token_type_ids: np.ndarray,
):
    """Test hdf5 text buffer write function to make sure it properly writes."""
    with tempfile.TemporaryDirectory() as output_dir:
        hdf5_file_path = os.path.join(output_dir, "temp.hdf5")
        with Hdf5FileBuffer(hdf5_file_path, max_seq_length1, DATA_TYPE, chunk_size) as f:
            for i in range(num_iterations):
                tokenized_line_copy = tokenized_line[:]
                tokenized_line_copy._token_ids = list(map(lambda x: x + i, tokenized_line_copy._token_ids))
                tokenized_line_copy._token_type_ids = list(map(lambda x: x + i, tokenized_line_copy._token_type_ids))
                f.write([tokenized_line_copy])
        with h5py.File(hdf5_file_path, "r") as f:
            assert str(f.keys()) == "<KeysViewHDF5 ['input_ids', 'token_type_ids']>"
            assert f["input_ids"].shape == gold_input_ids.shape
            for input_ids_i, gold_input_ids_i in zip(f["input_ids"], gold_input_ids):
                assert all(input_ids_i == gold_input_ids_i)
            assert f["token_type_ids"].shape == gold_token_type_ids.shape
            for token_type_ids_i, gold_token_type_ids_i in zip(f["token_type_ids"], gold_token_type_ids):
                assert all(token_type_ids_i == gold_token_type_ids_i)
