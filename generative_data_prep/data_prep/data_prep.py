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

import json
import os
import sys
from multiprocessing.managers import ValueProxy
from multiprocessing.synchronize import Lock
from typing import Dict, Optional

from transformers import PreTrainedTokenizerBase

from generative_data_prep.data_buffers import Hdf5FileBuffer
from generative_data_prep.processors import ArticleTokenizer
from generative_data_prep.utils import BoundaryType, FileExtension, PackingConfig


def data_prep_main(
    silent: bool,
    tokenizer: PreTrainedTokenizerBase,
    input_file: str,
    output_file: str,
    error_log_dir: str,
    max_seq_length: int,
    input_packing_config: PackingConfig,
    packing_boundary: BoundaryType,
    attention_boundary: BoundaryType,
    disable_space_separator: bool,
    keep_prompt_only_sequences: bool,
    ignore_input_format_error: bool,
    prompt_keyword: str,
    completion_keyword: str,
    num_skipped_articles: Optional[ValueProxy] = None,
    num_tokenized_articles: Optional[ValueProxy] = None,
    num_tokenized_articles_lock: Optional[Lock] = None,
    category_to_id: Optional[Dict[str, int]] = None,
    prompt_prefix: Optional[str] = None,
    prompt_postfix: Optional[str] = None,
    dataset_type: Optional[str] = None,
    apply_chat_template: Optional[bool] = False,
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
        num_tokenized_articles: Shared variable for number of tokenized articles.
        num_tokenized_articles_lock: Lock needed in order to updated shared variable.
        category_to_id: Dictionary that maps category string names to IDs.
        prompt_prefix: text to add before the prompt, for chatML conventions use.
        prompt_postfix: text to add after the prompt, for chatML conventions use.

    Returns:
        Metrics associated with tokenization
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
        category_to_id,
        prompt_prefix,
        prompt_postfix,
        apply_chat_template,
    )

    error_log_path = os.path.join(error_log_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}.log")

    dump_categories = category_to_id is not None

    with Hdf5FileBuffer(output_file, max_seq_length, dump_categories) as hdf5_text_buffer:
        with open(input_file, "r") as reader:
            for i, line in enumerate(reader):
                try:
                    hdf5_text_buffer.write(article_tokenizer(line))
                    if (
                        (i != 0 and i % 100 == 0)
                        and num_tokenized_articles_lock is not None
                        and num_tokenized_articles is not None
                    ):
                        with num_tokenized_articles_lock:
                            num_tokenized_articles.value += 100
                except json.JSONDecodeError as exc:
                    if ignore_input_format_error:
                        with open(error_log_path, "a") as f:
                            f.write(line)
                        if num_tokenized_articles_lock is not None and num_skipped_articles is not None:
                            with num_tokenized_articles_lock:
                                num_skipped_articles.value += 1
                        continue
                    else:
                        raise json.JSONDecodeError(
                            f"Error occurred loading this misformatted JSON line:\n\n{line}\n"
                            "Please format input dataset properly so that each line can be loaded with json.loads(). "
                            "Or consider using the --ignore_input_format_error flag to skip misformatted lines.",
                            exc.doc,
                            exc.pos,
                        ) from exc
            if num_tokenized_articles_lock is not None and num_tokenized_articles is not None:
                with num_tokenized_articles_lock:
                    num_tokenized_articles.value += i % 100
            hdf5_text_buffer.write(article_tokenizer(None))
    article_tokenizer.metrics.dataset_type = dataset_type
    return article_tokenizer.metrics
