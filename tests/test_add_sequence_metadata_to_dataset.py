import os
import tempfile

import h5py
import pytest
import yaml

from generative_data_prep.utils import (
    add_all_metadata_to_dataset,
    add_seq_metadata_dataset,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory and yield its path."""
    with tempfile.TemporaryDirectory() as test_dir:
        yield test_dir


@pytest.fixture
def temp_hdf5_files(temp_dir):
    """Create temporary HDF5 files for train and dev sets with input_ids datasets."""
    train_hdf5_path = os.path.join(temp_dir, "train_1.hdf5")
    dev_hdf5_path = os.path.join(temp_dir, "dev_1.hdf5")

    with h5py.File(train_hdf5_path, "w") as f:
        f.create_dataset("input_ids", data=[[1, 2, 3], [4, 5, 6]])

    with h5py.File(dev_hdf5_path, "w") as f:
        f.create_dataset("input_ids", data=[[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    return train_hdf5_path, dev_hdf5_path


@pytest.fixture
def temp_metadata_file(temp_dir):
    """Create a temporary metadata.yaml file."""
    metadata_path = os.path.join(temp_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.safe_dump({}, f)
    return metadata_path


def test_add_seq_metadata_dataset(temp_dir, temp_hdf5_files, temp_metadata_file):
    """Test adding sequence metadata to a dataset directory."""
    add_seq_metadata_dataset(temp_dir)

    with open(temp_metadata_file, "r") as f:
        metadata = yaml.safe_load(f)

    assert metadata["train_sequences"] == 2
    assert metadata["dev_sequences"] == 3


def test_add_all_metadata_to_dataset_from_example():
    """Test add_all_metadata_to_dataset on real example data."""
    dataset_path = "/Users/zoltanc/Desktop/generative_data_prep/tests/examples/metadata_test"
    expected_metadata = {
        "max_batch_size_dev": None,
        "max_batch_size_train": 1,
        "max_seq_length": 2048,
        "number_of_dev_files": 0,
        "number_of_test_files": 0,
        "number_of_training_files": 32,
        "token_type_ids": True,
        "tokenizer_model_type": None,
        "train_articles": 42,
        "train_completion_tokens": 1167,
        "train_input_tokens": None,
        "train_output_tokens": 65536,
        "train_padding_tokens": 62524,
        "train_prompt_tokens": 1845,
        "train_sequences": 32,
        "train_tokens_dropped_from_all_prompt": 0,
        "train_tokens_dropped_from_packing": 0,
        "vocab_size": None,
    }

    add_all_metadata_to_dataset(dataset_path)

    metadata_path = os.path.join(dataset_path, "metadata.yaml")
    with open(metadata_path, "r") as f:
        actual_metadata = yaml.safe_load(f)

    # Make sure all expected keys are present and values match
    for key, expected_value in expected_metadata.items():
        assert key in actual_metadata, f"Missing metadata key: {key}"
        assert (
            actual_metadata[key] == expected_value
        ), f"Mismatch for {key}: expected {expected_value}, got {actual_metadata[key]}"
