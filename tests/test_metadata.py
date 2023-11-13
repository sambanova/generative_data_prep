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
    context_dict = {
        "eval": False,
        "batch_size": 1,
        "model_type": str(type(BertConfig.from_pretrained("bert-base-uncased"))),
        "vocab_size": 50257,
        "world_size": 4,
        "max_seq_length": 1024,
    }
    try:
        DatasetMetadata.model_validate(metadata_dict, context=context_dict)
    except ValidationError as exc:
        print(exc, flush=True)
        assert True
        return
    assert False
