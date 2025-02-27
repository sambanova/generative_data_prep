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
from filecmp import cmpfiles
from glob import glob
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from generative_data_prep.utils import save_tokenizer


class generative_dataset(Dataset):
    """Object to reprent a generative dataset given an input hdf5 file."""

    def __init__(self, input_file: str):
        """Initialize a generative dataset."""
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        keys = ["input_ids", "token_type_ids"]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs[0])

    def __getitem__(self, index: int):
        """Overwride the git item function to just return token and type id."""
        [input_ids, token_type_ids] = [torch.from_numpy(input[index].astype(np.int32)) for input in self.inputs]
        return [input_ids.long(), token_type_ids.long()]


def check_diff_hdf5(file_1: str, file_2: str):
    """Check if two hdf5 output directories are the same."""
    ds1 = generative_dataset(file_1)
    ds2 = generative_dataset(file_2)

    assert len(ds1) == len(ds2), "Mismatched dataset length!"
    for i in range(len(ds1)):
        x1 = ds1[i]
        x2 = ds2[i]
        assert len(x1) == len(x2), "Mismatched example length!"
        for j in range(len(x1)):
            t1 = x1[j]
            t2 = x2[j]
            assert t1.shape == t2.shape, f"Mismatched example shape at index {j}, elt {i}!"
            assert torch.sum(t1 - t2).item() == 0, f"Mismatched IDs at index {j}, elt {i}!"


def check_splits(dir_1: str, dir_2: str):
    """Check to make sure that there is a split directory in both directories, and their contents are the same"""
    jsonl_path_dir_1 = os.path.join(dir_1, "splits")
    jsonl_path_dir_2 = os.path.join(dir_2, "splits")
    assert cmpfiles(jsonl_path_dir_1, jsonl_path_dir_2, os.listdir(jsonl_path_dir_1), shallow=False)


def check_no_split_dir(dir_1: str, dir_2: str):
    """Check to make sure that there is no splits directory in either [dir_1] or [dir_2]"""
    assert not os.path.exists(os.path.join(dir_1, "splits")) and not os.path.exists(os.path.join(dir_2, "splits"))


def check_pipeline(dir_1: str, dir_2: str):
    """Check if two output directories are the same."""
    assert os.listdir(dir_1).sort() == os.listdir(dir_2).sort() == ["test_files", "splits"].sort()

    test_path_dir1 = os.path.join(dir_1, "test_files")
    test_path_dir2 = os.path.join(dir_2, "test_files")
    if os.path.exists(test_path_dir1) or os.path.exists(test_path_dir1):
        assert cmpfiles(test_path_dir1, test_path_dir2, os.listdir(test_path_dir1), shallow=False)

    hdf5_files_1 = []
    for hdf5_file in os.listdir(dir_1):
        if ".hdf5" in hdf5_file:
            hdf5_files_1.append(hdf5_file)

    hdf5_files_2 = []
    for hdf5_file in os.listdir(dir_2):
        if ".hdf5" in hdf5_file:
            hdf5_files_2.append(hdf5_file)

    hdf5_files_1.sort()
    hdf5_files_2.sort()
    assert hdf5_files_1 == hdf5_files_2

    hdf5_files_1 = list(map(lambda x: os.path.join(dir_1, x), hdf5_files_1))
    hdf5_files_2 = list(map(lambda x: os.path.join(dir_2, x), hdf5_files_2))

    for hdf5_file_1, hdf5_file_2 in zip(hdf5_files_1, hdf5_files_2):
        check_diff_hdf5(hdf5_file_1, hdf5_file_2)


def check_balance(hdf5_dir: str, split: str = ""):
    """Asserts that the hdf5 files located under hdf5_dir/split*.hdf5
    all have the same number of sequences to within 1 sequence

    Args:
        hdf5_dir: directory that hdf5 files are located in
        split: name of split to balance (this will find all files under
        hdf5_dir and balance them)
    """
    hdf5_file_paths = glob(hdf5_dir + f"/{split}*.hdf5")
    input_id_shapes = []
    ttid_shapes = []

    for file_path in hdf5_file_paths:
        with h5py.File(file_path) as curr_hdf5_file:
            input_id_shapes.append(curr_hdf5_file["input_ids"].shape[0])
            ttid_shapes.append(curr_hdf5_file["token_type_ids"].shape[0])

    err_msg = f"hdf5 files under dir: {hdf5_dir} did not have the same number if sequences for input ids, "
    err_msg += f"largest hdf5 {max(input_id_shapes)}, smallest {min(input_id_shapes)}"
    assert abs(max(input_id_shapes) - min(input_id_shapes)) <= 1, err_msg
    err_msg = f"hdf5 files under dir: {ttid_shapes} did not have the same number if sequences for token type ids, "
    err_msg += f"largest hdf5 {max(input_id_shapes)}, smallest {min(ttid_shapes)}"
    assert abs(max(ttid_shapes) - min(ttid_shapes)) <= 1, err_msg


def test_save_tokenizer():
    """Tests save tokenizer for pretrained directory."""
    with TemporaryDirectory() as tokenizer_dir, TemporaryDirectory() as pretrained_dir:
        # Initialize a tokenizer and save it to the pretrained directory
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(pretrained_dir)

        # Call the function with the temporary directories
        save_tokenizer(tokenizer, tokenizer_dir, pretrained_dir)

        # Check if the tokenizer directory contains the expected files
        expected_files = set(os.listdir(pretrained_dir))
        saved_files = set(os.listdir(os.path.join(tokenizer_dir, "user_input_tokenizer")))
        assert expected_files.intersection(saved_files) == expected_files, "Not all files were copied correctly."
        new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        new_tokenizer.encode("hello!")
        assert True


def test_save_tokenizer_with_huggingface_id():
    """Tests save tokenizer for pretrained directory"""
    with TemporaryDirectory() as tokenizer_dir:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Call the function with a Hugging Face model ID
        save_tokenizer(tokenizer, tokenizer_dir, model_name)

        # Check if the Hugging Face model ID file exists
        hf_model_id_file = os.path.join(tokenizer_dir, "user_input_tokenizer", "huggingface_model_id.txt")
        assert os.path.exists(hf_model_id_file), "Hugging Face model ID file is missing."

        # Verify the content of the file
        with open(hf_model_id_file, "r") as f:
            assert f.read().strip() == model_name, "Hugging Face model ID does not match."
