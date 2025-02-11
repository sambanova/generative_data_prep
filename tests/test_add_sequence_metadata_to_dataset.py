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
def temp_hdf5_file(temp_dir):
    """Create a temporary HDF5 file with an input_ids dataset."""
    hdf5_path = os.path.join(temp_dir, "test.h5")
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("input_ids", data=[[1, 2, 3], [4, 5, 6]])
    return hdf5_path


@pytest.fixture
def temp_metadata_file(temp_dir):
    """Create a temporary metadata.yaml file."""
    metadata_path = os.path.join(temp_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.safe_dump({}, f)
    return metadata_path


def test_add_seq_metadata_dataset(temp_dir, temp_hdf5_file, temp_metadata_file):
    """Test adding sequence metadata to a dataset directory."""
    add_seq_metadata_dataset(temp_dir)
    with open(temp_metadata_file, "r") as f:
        metadata = yaml.safe_load(f)
    assert metadata["sequences"] == 2
