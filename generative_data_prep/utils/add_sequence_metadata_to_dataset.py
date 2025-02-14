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

This module allows you to update existing datasets with number of sequences metadata.
"""

import os

import h5py
import yaml


def count_sequences_in_hdf5(file_path):
    """Count the total number of sequences in an HDF5 file."""
    total_sequences = 0
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            data = hdf5_file["input_ids"]
            if len(data.shape) > 1:  # Ensure it has at least two dimensions
                total_sequences += data.shape[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return total_sequences


def update_metadata(metadata, key, total_sequences):
    """Update the metadata dictionary with the sequence count."""
    metadata[key] = total_sequences
    return metadata


def save_metadata(metadata_path, metadata):
    """Save the metadata dictionary to the metadata file."""
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False)


def add_seq_metadata_dataset(dataset_path):
    """Iterate through subdirectories, count sequences in HDF5 files, and update metadata."""
    metadata_path = os.path.join(dataset_path, "metadata.yaml")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading metadata {metadata_path}: {e}")

    train_sequences = 0
    dev_sequences = 0

    for file in os.listdir(dataset_path):
        if file.endswith(".h5") or file.endswith(".hdf5"):
            file_path = os.path.join(dataset_path, file)
            total_sequences = count_sequences_in_hdf5(file_path)
            if "train" in file.lower():
                train_sequences += total_sequences
            elif "dev" in file.lower():
                dev_sequences += total_sequences

    if train_sequences > 0:
        metadata = update_metadata(metadata, "train_sequences", train_sequences)
    if dev_sequences > 0:
        metadata = update_metadata(metadata, "dev_sequences", dev_sequences)

    save_metadata(metadata_path, metadata)


def add_seq_metadata_to_dir_of_datasets(root_dir):
    """Iterate through subdirectories, count sequences in HDF5 files, and update metadata."""
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            add_seq_metadata_dataset(subdir_path)
