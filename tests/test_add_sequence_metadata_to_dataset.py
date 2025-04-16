import os
import tempfile
from typing import Dict

import h5py
import pytest
import yaml
from transformers import AutoTokenizer, GPT2Config, PreTrainedTokenizerBase

from generative_data_prep.data_prep import pipeline_main
from generative_data_prep.utils import (
    BoundaryType,
    PackingConfig,
    add_all_metadata_to_dataset,
    add_seq_metadata_dataset,
)
from tests.conftest import TESTS_EXAMPLES_PATH

LLAMA_TOKENIZER = AutoTokenizer.from_pretrained("arcee-ai/Llama-3.1-SuperNova-Lite")


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


def get_input_path(test_name: str) -> str:
    """Create an absolute path to an example input."""
    base_path = TESTS_EXAMPLES_PATH / test_name / f"example_{test_name}_data"
    if os.path.isdir(base_path):
        return base_path
    else:
        ext = ".txt" if "txt" in test_name else ".jsonl"
        return f"{base_path}{ext}"


MODEL_CONFIG = GPT2Config.from_pretrained("gpt2")


@pytest.mark.parametrize(
    "test_name,disable_space_separator,keep_prompt_only_sequences,ignore_input_format_error,\
    prompt_keyword,completion_keyword,shuffle,do_not_balance_hdf5,keep_split_jsonls,max_seq_length,\
    input_packing_config,packing_boundary,attention_boundary,num_training_splits,num_dev_splits,\
    num_test_splits,category_to_id,dev_ratio,test_ratio,tokenizer,apply_chat_template",
    [
        (
            "apply_chat_template",
            False,
            True,
            False,
            "prompt",
            "completion",
            "False",
            False,
            True,
            1024,
            PackingConfig.from_str("greedy::drop"),
            BoundaryType.JSONL,
            BoundaryType.JSONL,
            16,
            0,
            0,
            None,
            None,
            None,
            LLAMA_TOKENIZER,
            True,
        )
    ],
)
def test_add_all_metadata_to_dataset_reproducibility(
    test_name: str,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    ignore_input_format_error: bool,
    prompt_keyword: str,
    completion_keyword: str,
    shuffle: str,
    do_not_balance_hdf5: bool,
    keep_split_jsonls: bool,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    num_training_splits: int,
    num_dev_splits: int,
    num_test_splits: int,
    category_to_id: Dict[str, int],
    dev_ratio: float,
    test_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    apply_chat_template: bool,
):
    """Test that `add_all_metadata_to_dataset` produces the same metadata as the original pipeline."""
    num_workers = os.cpu_count() or 1
    input_path = get_input_path(test_name)

    with tempfile.TemporaryDirectory() as output_dir:
        # Run the pipeline to generate dataset
        pipeline_main(
            input_path=input_path,
            tokenizer=tokenizer,
            pretrained_tokenizer="gpt2",
            model_config=MODEL_CONFIG,
            output_dir=output_dir,
            disable_space_separator=disable_space_separator,
            keep_prompt_only_sequences=keep_prompt_only_sequences,
            ignore_input_format_error=ignore_input_format_error,
            prompt_keyword=prompt_keyword,
            completion_keyword=completion_keyword,
            shuffle=shuffle,
            overwrite_output_path=False,
            num_workers=num_workers,
            do_not_balance_hdf5=do_not_balance_hdf5,
            keep_split_jsonls=keep_split_jsonls,
            max_seq_length=max_seq_length,
            input_packing_config=input_packing_config,
            packing_boundary=packing_boundary,
            attention_boundary=attention_boundary,
            num_training_splits=num_training_splits,
            num_dev_splits=num_dev_splits,
            num_test_splits=num_test_splits,
            category_to_id=category_to_id,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            apply_chat_template=apply_chat_template,
        )

        metadata_path = os.path.join(output_dir, "metadata.yaml")
        assert os.path.exists(metadata_path), "Original metadata.yaml not found after pipeline run."

        # Load the original metadata
        with open(metadata_path, "r") as f:
            original_metadata = yaml.safe_load(f)

        # Rerun metadata collection
        add_all_metadata_to_dataset(output_dir)

        # Reload and compare
        with open(metadata_path, "r") as f:
            regenerated_metadata = yaml.safe_load(f)

    IGNORE_KEYS = [
        "tokenizer_model_type",
        "train_tokens_dropped_from_all_prompt",
        "train_tokens_dropped_from_packing",
        "vocab_size",
        "train_input_tokens",
    ]
    for k, v in original_metadata.items():
        if k not in IGNORE_KEYS:
            assert (
                v == regenerated_metadata[k]
            ), f"key {k}, Value in original_metadata {v}, Value in regenerated_metadata {regenerated_metadata[k]}"
