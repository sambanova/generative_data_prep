import os
from pathlib import Path

import yaml
from pydantic import ValidationError
from transformers import BertConfig, GPT2Config

from generative_data_prep.utils import DatasetMetadata


def test_pydantic_model_passing():
    """Testing DatasetMetadata loads in variables correctly. This should pass"""
    output_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(GPT2Config.from_pretrained("gpt2"))),
        "vocab_size": 50257,
        "world_size": 4,
        "max_seq_length": 1024,
    }
    DatasetMetadata.model_validate(metadata_dict, context=context_dict)


def test_pydantic_model_wrong_model_type():
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    bert_config = BertConfig.from_pretrained("bert-base-uncased")
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(bert_config)),
        "vocab_size": bert_config.vocab_size,
        "world_size": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 1
        error = exc.errors()[0]
        if "tokenizer_model_type" == error["loc"][0]:
            assert "BertConfig" in error["msg"]
            assert "GPT2Config" in error["msg"]
            assert "does not match model type used during training" in error["msg"]
        else:
            assert False
        return
    assert False


def test_pydantic_model_greater_vocab_size():
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(GPT2Config.from_pretrained("gpt2"))),
        "vocab_size": 60000,
        "world_size": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        assert len(exc.errors()) == 1
        print(exc.errors(), flush=True)
        error = exc.errors()[0]
        if "vocab_size" == error["loc"][0]:
            assert "60000" in error["msg"]
            assert "50257" in error["msg"]
            assert "exceeds the maximum allowed batch size" in error["msg"]
        else:
            assert False
        return
    assert False


def test_pydantic_model_greater_world_size():
    """Testing DatasetMetadata loads in variables correctly. This should fail"""
    output_dir = os.path.join(
        Path.cwd(), "tests", "examples", "pretraining_sha256_split", "pipelined_pretraining_sha256_split"
    )
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "r") as file:
        metadata_dict = yaml.safe_load(file)
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(GPT2Config.from_pretrained("gpt2"))),
        "vocab_size": 60000,
        "world_size": 100,
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
            assert "greater than" in error["msg"]
        else:
            assert False
        return
    assert False
