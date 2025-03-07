import os
import tempfile

import h5py
import pytest
import yaml

from generative_data_prep.utils import add_seq_metadata_dataset


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
