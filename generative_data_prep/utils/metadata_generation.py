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

from pydantic import BaseModel


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
    model_type: str
    number_of_training_files: int
    number_of_dev_files: int
    number_of_test_files: int
    max_batch_size_train: int
    max_batch_size_dev: int
