import logging
import os
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from generative_data_prep.__main__ import main
from generative_data_prep.utils import BoundaryType, PackingConfig, validate_sha256


@pytest.fixture(scope="session")
def shared_output_dir_with_split(tmp_path_factory):
    """Creating temp directories for each test and running generative pipeline e2e"""
    tmp_split_output_dir = tmp_path_factory.mktemp("pretraining_sha256_split")
    logging.info(f"temporary split output directory is in {tmp_split_output_dir}")

    input_file = os.path.join(Path.cwd(), "tests", "examples", "pretraining", "example_pretraining_data.jsonl")
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    args_dict = {
        "cmd": "pipeline",
        "num_training_splits": 4,
        "dev_ratio": None,
        "num_dev_splits": None,
        "test_ratio": None,
        "num_test_splits": None,
        "shuffle": "on_RAM",
        "num_workers": num_workers,
        "keep_split_jsonls": True,
        "input_file_path": input_file,
        "output_path": tmp_split_output_dir,
        "overwrite_output_path": False,
        "disable_space_separator": False,
        "keep_prompt_only_sequences": False,
        "silent": False,
        "do_not_balance_hdf5": False,
        "log_file_path": None,
        "tokenizer_class": None,
        "pretrained_tokenizer": None,
        "vocab_file": None,
        "merges_file": None,
        "max_seq_length": 1024,
        "input_packing_config": PackingConfig.get_default(),
        "packing_boundary": BoundaryType.JSONL.value,
        "attention_boundary": BoundaryType.JSONL.value,
        "special_tokens_dict": None,
        "prompt_keyword": "prompt",
        "completion_keyword": "completion",
        "prompt_prefix": None,
        "prompt_postfix": None,
        "categories_path": None,
    }
    args = Namespace(**args_dict)
    main(args)

    return tmp_split_output_dir


def test_validation_sha_with_split(shared_output_dir_with_split):
    """Testing validation method and sha256 created correctly. With splits."""

    assert validate_sha256(shared_output_dir_with_split)


def fake_getsize(file_path):
    """Fake get size function for monkeypatch."""
    return 0


@pytest.fixture
def mock_getsize(monkeypatch):
    """Mocking size function using monkeypatch."""
    monkeypatch.setattr(os.path, "getsize", fake_getsize)


def test_validation_sha_with_split_redoing_sha256(mock_getsize, shared_output_dir_with_split):
    """Testing validation method and sha256 created correctly. With splits."""

    assert validate_sha256(shared_output_dir_with_split)


def test_validation_sha_with_split_and_eval(tmp_path):
    """Testing validation method and sha256 created correctly. With splits."""

    tmp_split_with_eval_output_dir = tmp_path / "pretraining_sha256_split_and_eval"
    logging.info(f"temporary split with eval output directory is in {tmp_split_with_eval_output_dir}")

    input_file = os.path.join(Path.cwd(), "tests", "examples", "pretraining", "example_pretraining_data.jsonl")

    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    args_dict = {
        "cmd": "pipeline",
        "num_training_splits": 4,
        "dev_ratio": None,
        "num_dev_splits": 1,
        "test_ratio": None,
        "num_test_splits": 1,
        "shuffle": "on_RAM",
        "num_workers": num_workers,
        "keep_split_jsonls": True,
        "input_file_path": input_file,
        "output_path": tmp_split_with_eval_output_dir,
        "overwrite_output_path": False,
        "disable_space_separator": False,
        "keep_prompt_only_sequences": False,
        "silent": False,
        "do_not_balance_hdf5": False,
        "log_file_path": None,
        "tokenizer_class": None,
        "pretrained_tokenizer": None,
        "vocab_file": None,
        "merges_file": None,
        "max_seq_length": 1024,
        "input_packing_config": PackingConfig.get_default(),
        "packing_boundary": BoundaryType.JSONL.value,
        "attention_boundary": BoundaryType.JSONL.value,
        "special_tokens_dict": None,
        "prompt_keyword": "prompt",
        "completion_keyword": "completion",
        "prompt_prefix": None,
        "prompt_postfix": None,
        "categories_path": None,
    }
    args = Namespace(**args_dict)
    main(args)

    assert validate_sha256(tmp_split_with_eval_output_dir)


def test_validation_sha_without_split(tmp_path):
    """Testing validation method and sha256 created correctly. Without splits."""

    tmp_no_split_output_dir = tmp_path / "pretraining_sha256_no_split"
    logging.info(f"temporary no split output directory is in {tmp_no_split_output_dir}")

    input_file = os.path.join(Path.cwd(), "tests", "examples", "pretraining", "example_pretraining_data.jsonl")

    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    args_dict = {
        "cmd": "pipeline",
        "num_training_splits": 4,
        "dev_ratio": None,
        "num_dev_splits": None,
        "test_ratio": None,
        "num_test_splits": None,
        "shuffle": "on_RAM",
        "num_workers": num_workers,
        "keep_split_jsonls": False,
        "input_file_path": input_file,
        "output_path": tmp_no_split_output_dir,
        "overwrite_output_path": False,
        "disable_space_separator": False,
        "keep_prompt_only_sequences": False,
        "silent": False,
        "do_not_balance_hdf5": False,
        "log_file_path": None,
        "tokenizer_class": None,
        "pretrained_tokenizer": None,
        "vocab_file": None,
        "merges_file": None,
        "max_seq_length": 1024,
        "input_packing_config": PackingConfig.get_default(),
        "packing_boundary": BoundaryType.JSONL.value,
        "attention_boundary": BoundaryType.JSONL.value,
        "special_tokens_dict": None,
        "prompt_keyword": "prompt",
        "completion_keyword": "completion",
        "prompt_prefix": None,
        "prompt_postfix": None,
        "categories_path": None,
    }
    args = Namespace(**args_dict)
    main(args)

    assert validate_sha256(tmp_no_split_output_dir)


def test_validation_corrupted(shared_output_dir_with_split, tmp_path):
    """Testing validation method and sha256 created correctly. Without splits."""

    tmp_corrupted_no_split_output_dir = tmp_path / "corrupted_pretraining_sha256_no_split"
    logging.info(f"temporary split corrupted output directory is in {tmp_corrupted_no_split_output_dir}")

    shutil.copytree(shared_output_dir_with_split, tmp_corrupted_no_split_output_dir)
    tmp_corrupted_metadata = tmp_corrupted_no_split_output_dir / "metadata.yaml"
    with open(tmp_corrupted_metadata, "r") as file:
        metadata_dict = yaml.safe_load(file)
    metadata_dict["vocab_size"] = 0
    with open(tmp_corrupted_metadata, "w") as file:
        yaml.dump(metadata_dict, file, default_flow_style=False)

    assert not validate_sha256(tmp_corrupted_no_split_output_dir)
