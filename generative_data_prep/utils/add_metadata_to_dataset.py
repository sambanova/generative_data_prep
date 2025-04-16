"""Module for adding metadata to HDF5 datasets used in generative model training.

This includes counting sequences, token types, and updating metadata files
accordingly for each dataset directory.
"""

import os

import h5py
import numpy as np
import yaml
from transformers import AutoConfig, AutoTokenizer

from .constants import TokenTypeIds

METADATA_KEYS_CANT_ADD = [
    "train_tokens_dropped_from_all_prompt",
    "train_tokens_dropped_from_packing",
    "train_input_tokens",
]


def count_sequences_in_hdf5(file_path):
    """Count the total number of sequences in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        int: Total number of sequences.
    """
    total_sequences = 0
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            data = hdf5_file["input_ids"]
            if len(data.shape) > 1:
                total_sequences += data.shape[0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return total_sequences


def update_metadata(metadata, key, total_sequences):
    """Update the metadata dictionary with a sequence count.

    Args:
        metadata (dict): Metadata dictionary to update.
        key (str): Metadata key to update.
        total_sequences (int): Number of sequences to add.

    Returns:
        dict: Updated metadata dictionary.
    """
    metadata[key] = total_sequences
    return metadata


def save_metadata(metadata_path, metadata):
    """Save metadata dictionary to a YAML file.

    Args:
        metadata_path (str): Path to the metadata YAML file.
        metadata (dict): Metadata dictionary to save.
    """
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False)


def add_seq_metadata_dataset(dataset_path):
    """Add train/dev sequence count metadata to a dataset directory.

    Args:
        dataset_path (str): Path to the dataset directory.
    """
    metadata_path = os.path.join(dataset_path, "metadata.yaml")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f) or {}

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
    """Apply `add_seq_metadata_dataset` to each subdirectory in a directory.

    Args:
        root_dir (str): Root directory containing dataset subdirectories.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            add_seq_metadata_dataset(subdir_path)


def add_all_metadata_to_dataset(dataset_path):  # noqa: C901
    """Add detailed token and sequence metadata to a dataset.

    This includes token counts for prompts, completions, padding, articles, etc.,
    as well as batch size and sequence length statistics.

    Args:
        dataset_path (str): Path to the dataset directory.
    """
    metadata = {}
    tokenizer_dir = os.path.join(dataset_path, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        tokenizer_dir = dataset_path

    metadata["tokenizer_model_type"] = None
    metadata["vocab_size"] = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        metadata["vocab_size"] = tokenizer.vocab_size
        config = AutoConfig.from_pretrained(tokenizer.name_or_path)
        metadata["tokenizer_model_type"] = str(type(config))
    except Exception as e:
        print(f"Unable to add tokenizer related metadata because of error {e}")

    train_sequences = 0
    train_completion_tokens = 0
    train_prompt_tokens = 0
    train_articles = 0
    train_padding_tokens = 0
    train_output_tokens = 0
    max_seq_length = None
    max_batch_size_train = None
    max_batch_size_dev = None

    number_of_training_files = 0
    number_of_dev_files = 0
    number_of_test_files = 0

    for file in os.listdir(dataset_path):
        if not (file.endswith(".h5") or file.endswith(".hdf5")):
            continue

        file_path = os.path.join(dataset_path, file)

        try:
            with h5py.File(file_path, "r") as hdf5_file:
                input_ids = hdf5_file["input_ids"]
                token_type_ids = hdf5_file["token_type_ids"]
                shape = input_ids.shape

                file_sequences = shape[0]
                file_tokens = shape[0] * shape[1]
                if "train" in file.lower():
                    number_of_training_files += 1
                    train_sequences += file_sequences
                    train_output_tokens += file_tokens

                    train_completion_tokens += np.sum(token_type_ids[:] == TokenTypeIds.COMPLETION)
                    train_prompt_tokens += np.sum(token_type_ids[:] == TokenTypeIds.PROMPT)
                    train_articles += np.sum(token_type_ids[:] == TokenTypeIds.SEP)
                    train_padding_tokens += np.sum(token_type_ids[:] == TokenTypeIds.PADDING)

                    if max_batch_size_train is None or file_sequences < max_batch_size_train:
                        max_batch_size_train = file_sequences
                    if max_seq_length is None:
                        max_seq_length = shape[1]

                elif "dev" in file.lower():
                    number_of_dev_files += 1
                    if max_batch_size_dev is None or file_sequences < max_batch_size_dev:
                        max_batch_size_dev = file_sequences
                elif "test" in file.lower():
                    number_of_test_files += 1

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    metadata.update(
        {
            "train_articles": int(train_articles),
            "train_completion_tokens": int(train_completion_tokens + train_articles),
            "train_input_tokens": None,
            "max_batch_size_dev": int(max_batch_size_dev) if max_batch_size_dev is not None else None,
            "max_batch_size_train": int(max_batch_size_train) if max_batch_size_train is not None else None,
            "max_seq_length": int(max_seq_length) if max_seq_length is not None else None,
            "number_of_dev_files": number_of_dev_files,
            "number_of_test_files": number_of_test_files,
            "number_of_training_files": number_of_training_files,
            "train_output_tokens": int(train_output_tokens),
            "train_padding_tokens": int(train_padding_tokens),
            "train_prompt_tokens": int(train_prompt_tokens),
            "train_sequences": int(train_sequences),
            "token_type_ids": True,
            "train_tokens_dropped_from_all_prompt": None,
            "train_tokens_dropped_from_packing": None,
        }
    )

    save_metadata(os.path.join(dataset_path, "metadata.yaml"), metadata)


def add_all_metadata_to_dir_of_datasets(root_dir):
    """Apply `add_all_metadata_to_dataset` to each dataset in a directory.

    Args:
        root_dir (str): Root directory containing dataset subdirectories.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            add_all_metadata_to_dataset(subdir_path)
