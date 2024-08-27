import logging
import os
from argparse import Namespace

import pytest
import yaml
from pydantic import ValidationError
from transformers import AutoConfig

from generative_data_prep.__main__ import main
from generative_data_prep.utils import BoundaryType, DatasetMetadata, PackingConfig
from tests.conftest import TESTS_EXAMPLES_PATH

PRETRAINING_SPLIT_METADATA_SHA256_PATH = TESTS_EXAMPLES_PATH / "pretraining_split_with_new_metadata_and_sha256"


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_passing(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should pass"""
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "vocab_size": 50257,
        "use_token_type_ids": use_token_type_ids,
        "number_of_workers": 4,
        "max_seq_length": 1024,
    }
    DatasetMetadata.model_validate(metadata_dict, context=context_dict)


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_wrong_model_type_and_less_vocab_size(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    error_keys = ["tokenizer_model_type", "vocab_size"]
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    bert_config = AutoConfig.from_pretrained("bert-base-uncased")
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(bert_config)),
        "vocab_size": bert_config.vocab_size,
        "use_token_type_ids": use_token_type_ids,
        "number_of_workers": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 2
        for error in exc.errors():
            assert error["loc"][0] in error_keys
            if "tokenizer_model_type" == error["loc"][0]:
                assert "BertConfig" in error["msg"]
                assert "GPT2Config" in error["msg"]
                assert "does not match model type used during training" in error["msg"]
            elif "vocab_size" == error["loc"][0]:
                assert "30522" in error["msg"]
                assert "50257" in error["msg"]
                assert "must be equal to or greater than dataset vocab size" in error["msg"]
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_greater_world_size(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "use_token_type_ids": use_token_type_ids,
        "vocab_size": 50257,
        "number_of_workers": 100,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 1
        error = exc.errors()[0]
        if "number_of_training_files" == error["loc"][0]:
            assert "100" in error["msg"]
            assert "4" in error["msg"]
            assert "is greater than the specified number of files" in error["msg"]
        else:
            assert False
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_different_sequence_length(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "use_token_type_ids": use_token_type_ids,
        "vocab_size": 50257,
        "number_of_workers": 4,
        "max_seq_length": 2048,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 1
        error = exc.errors()[0]
        if "max_seq_length" == error["loc"][0]:
            assert "2048" in error["msg"]
            assert "1024" in error["msg"]
            assert "must match max_seq_length" in error["msg"]
        else:
            assert False
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_greater_batch_size(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 30,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "use_token_type_ids": use_token_type_ids,
        "vocab_size": 50257,
        "number_of_workers": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 1
        error = exc.errors()[0]
        if "max_batch_size_train" == error["loc"][0]:
            assert "30" in error["msg"]
            assert "13" in error["msg"]
            assert "exceeds the maximum allowed batch size" in error["msg"]
        else:
            assert False
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_no_evaluation_files(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    error_keys = ["number_of_dev_files", "max_batch_size_dev"]
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": True,
        "batch_size": 1,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "use_token_type_ids": use_token_type_ids,
        "vocab_size": 50257,
        "number_of_workers": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 2
        for error in exc.errors():
            assert error["loc"][0] in error_keys
            if error["loc"][0] == "max_batch_size_dev":
                assert (
                    "Evaluation during training is turned on but there are no " "evaluation files in this dataset"
                ) in error["msg"]
            elif error["loc"][0] == "number_of_dev_files":
                assert ("Evaluating during runtime but have no evaluation files to run in dataset") in error["msg"]
            else:
                assert False
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_yes_evaluation_files(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should pass"""
    output_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    metadata_dict["number_of_dev_files"] = 1
    metadata_dict["max_batch_size_dev"] = 7
    context_dict = {
        "eval": True,
        "batch_size": 1,
        "model_type": str(type(AutoConfig.from_pretrained("gpt2"))),
        "use_token_type_ids": use_token_type_ids,
        "vocab_size": 50257,
        "number_of_workers": 4,
        "max_seq_length": 1024,
    }
    DatasetMetadata.model_validate(metadata_dict, context=context_dict)


def test_metadata_end2end_output(tmp_path):
    """Testing the e2e pipeline is still the same. This should pass"""

    tmp_e2e_output_dir = tmp_path / "e2e_testing_output_directory"
    logging.info(f"temporary e2e output directory is in {tmp_e2e_output_dir}")
    base_dir = PRETRAINING_SPLIT_METADATA_SHA256_PATH
    input_file = os.path.join(base_dir, "example_pretraining_data.jsonl")
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
        "ignore_input_format_error": False,
        "input_file_path": input_file,
        "output_path": tmp_e2e_output_dir,
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
        "apply_chat_template": False,
    }
    args = Namespace(**args_dict)
    main(args)
    metadata_file = os.path.join(tmp_e2e_output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    ground_truth_metadata_file = os.path.join(base_dir, "metadata.yaml")
    with open(ground_truth_metadata_file, "r") as file:
        ground_truth_metadata_dict = yaml.safe_load(file)

    assert metadata_dict == ground_truth_metadata_dict
