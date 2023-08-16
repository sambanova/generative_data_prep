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


Endpoint for tokenizing a jsonl file into a jsonl file.
"""

from __future__ import absolute_import

import os
import sys
from typing import Optional

from transformers import PreTrainedTokenizerBase

from generative_data_prep.data_buffers import Hdf5FileBuffer
from generative_data_prep.processors import ArticleTokenizer
from generative_data_prep.utils import BoundaryType, FileExtension, PackingConfig


def data_prep_main(
    silent: bool,
    tokenizer: PreTrainedTokenizerBase,
    input_file: str,
    output_file: str,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    prompt_keyword: str,
    completion_keyword: str,
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
):
    """Tokenize input_file into packed sequences stored in output_file.

    Args:
        silent: Whether to print or not
        tokenizer: tokenizer to call tokenizer.encode for tokenizing.
        input_file: Input jsonl file to tokenize.
        output_file: Tokenized output hdf5 file.
        max_seq_length: Maximum number of tokens that fit in models sequence.
        input_packing_config: How to pack the inputs when doing tokenization.
        packing_boundary: How to define the boundary when packing.
        attention_boundary: How to define the boundary when attending to other tokens.
        disable_space_separator: Disable adding space separator if true.
        keep_prompt_only_sequences: Keep sequences that only have prompt tokens if true.
        prompt_keyword: Prompt keyword to use as key in jsonl.
        completion_keyword: Completion keyword to use as key in jsonl.
        disable_space_separator: Disable adding space separator if true.
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.
    """
    if silent:
        sys.stdout = open(os.devnull, "w")

    file_ext = FileExtension(os.path.splitext(input_file)[1])
    article_tokenizer = ArticleTokenizer(
        tokenizer,
        max_seq_length,
        file_ext,
        input_packing_config,
        packing_boundary,
        attention_boundary,
        disable_space_separator,
        keep_prompt_only_sequences,
        prompt_keyword,
        completion_keyword,
        prompt_prefix,
        prompt_postfix,
    )

    with Hdf5FileBuffer(output_file, max_seq_length) as hdf5_text_buffer:
        with open(input_file, "r") as reader:
            for line in reader:
                hdf5_text_buffer.write(article_tokenizer(line))
            hdf5_text_buffer.write(article_tokenizer(None))
