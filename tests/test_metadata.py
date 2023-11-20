import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from transformers import AutoConfig

from generative_data_prep.utils import DatasetMetadata


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_passing(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should pass"""
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
            assert (
                "Evaluation during training is turned on but there are no " "evaluation files in this dataset"
            ) in error["msg"]
        return
    assert False


@pytest.mark.parametrize("use_token_type_ids", [(True), (False)])
def test_pydantic_model_yes_evaluation_files(use_token_type_ids):
    """Testing DatasetMetadata loads in variables correctly. This should pass"""
    output_dir = os.path.join(
        Path.cwd(),
        "tests",
        "examples",
        "pretraining_split_with_new_metadata_and_sha256",
        "pipelined_pretraining_sha256_split",
    )
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
