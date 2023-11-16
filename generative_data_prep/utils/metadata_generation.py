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


DatasetMetadata pydantic model and functions to verify pydantic model.
"""
from typing import Optional

from pydantic import BaseModel, ValidationInfo, field_validator


class DatasetMetadata(BaseModel):
    """Pydantic model to validate parameters for metadata.

    Args:
        max_seq_length: max sequence length of model
        token_type_ids: whether or not contains token type ids (always true)
        vocab_size: vocab size of tokenizer
        model_type: model type used for tokenizer
        number_of_training_files: number of hdf5 files available for training
        number_of_dev_files: number of hdf5 files available for evaluation during training
        number_of_test_files: number of test files
        max_batch_size_train: the minimum datapoints across all training file
        which is what the maximum batch size during training can be
        max_batch_size_dev: the minimum datapoints across all dev file which
        is what the maximum batch size during training can be

    """

    max_seq_length: int
    token_type_ids: bool
    vocab_size: int
    tokenizer_model_type: str
    number_of_training_files: int
    number_of_dev_files: Optional[int]
    number_of_test_files: Optional[int]
    max_batch_size_train: int
    max_batch_size_dev: Optional[int]

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, v: int, info: ValidationInfo):
        """Validates max sequence length."""
        runtime_max_seq_length = info.context.get("max_seq_length")
        if type(runtime_max_seq_length) is not int:
            raise ValueError("Max sequence length context param should be an integer variable")
        if v != runtime_max_seq_length:
            raise ValueError(
                (
                    f"max_seq_length specified during training ({runtime_max_seq_length}) "
                    f"must match max_seq_length used during generative data prep ({v})"
                )
            )
        return v

    @field_validator("token_type_ids")
    @classmethod
    def validate_token_type_ids(cls, v: bool, info: ValidationInfo):
        """Validates number of training files."""
        use_token_type_ids = info.context.get("use_token_type_ids")
        if type(use_token_type_ids) is not bool:
            raise ValueError("Token type id context param should be an boolean variable")
        if use_token_type_ids is True:
            if v is not True:
                raise ValueError(
                    "Token type ids is required for training however it is not a key within the datapoints"
                )
        return v

    @field_validator("number_of_training_files")
    @classmethod
    def validate_number_of_training_files(cls, v: int, info: ValidationInfo):
        """Validates number of training files."""
        number_of_workers = info.context.get("number_of_workers")
        if type(number_of_workers) is not int:
            raise ValueError("World size context param should be an integer variable")
        if v < number_of_workers:
            raise ValueError(
                (f"The number of workers({number_of_workers}) is greater than " f"the specified number of files ({v})")
            )
        return v

    @field_validator("vocab_size")
    @classmethod
    def validation_vocab_size(cls, v: int, info: ValidationInfo):
        """Validates vocab size."""
        runtime_vocab_size = info.context.get("vocab_size")
        if type(runtime_vocab_size) is not int:
            raise ValueError("Vocab size context param should be an integer variable")
        if v > runtime_vocab_size:
            raise ValueError(
                f"Runtime vocab size ({runtime_vocab_size}) must be equal to or greater than dataset vocab size ({v})"
            )
        return v

    @field_validator("tokenizer_model_type")
    @classmethod
    def validation_tokenizer_model_type(cls, v: str, info: ValidationInfo):
        """Validates model type."""
        str_model_type = info.context.get("model_type")
        if type(str_model_type) is not str:
            raise ValueError("Model type context param should be the type(model_config) and then passed in as a string")
        if v != str_model_type:
            raise ValueError(
                (
                    f"Model type of model during runtime ({str_model_type}) "
                    f"does not match model type used during training ({v})"
                )
            )

    @field_validator("number_of_dev_files")
    @classmethod
    def validation_number_of_dev_files(cls, v: int, info: ValidationInfo):
        """Validates number of dev files."""
        do_eval = info.context.get("eval")
        if type(do_eval) is not bool:
            raise ValueError("eval context param should be a boolean variable")
        if do_eval:
            if v is None:
                raise ValueError(
                    "Evaluation during training is turned on but there are no evaluation files in this dataset"
                )
            if v == 0:
                raise ValueError("Evaluating during runtime but have no evaluation files to run in dataset")

    @field_validator("max_batch_size_train")
    @classmethod
    def validation_batch_size_train(cls, v: int, info: ValidationInfo):
        """Validates batch size for training."""
        runtime_batch_size = info.context.get("batch_size")
        if type(runtime_batch_size) is not int:
            raise ValueError("batch_size context param should be an integer variable")
        if runtime_batch_size > v:
            raise ValueError(
                (
                    f"batch size specified during training ({runtime_batch_size}) exceeds the maximum "
                    f"allowed batch size ({v}) based on training dataset"
                )
            )

    @field_validator("max_batch_size_dev")
    @classmethod
    def validation_batch_size_dev(cls, v: int, info: ValidationInfo):
        """Validates bath size for evaluation."""
        do_eval = info.context.get("eval")
        runtime_batch_size = info.context.get("batch_size")
        if type(do_eval) is not bool:
            raise ValueError("eval context param should be a boolean variable")
        if do_eval:
            if v is None:
                raise ValueError(
                    "Evaluation during training is turned on but there are no evaluation files in this dataset"
                )
            if runtime_batch_size > v:
                raise ValueError(
                    (
                        f"batch size specified during training ({runtime_batch_size}) exceeds the maximum "
                        f"allowed batch size ({v}) based on evaluation files in dataset"
                    )
                )
