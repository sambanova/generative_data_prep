"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This module implements ArticleTokenizer.

Article tokenizer takes any input text or jsonl, runs tokenization using
the passed in tokenizer, and then packs the tokens into sequences using
the SequencePacker class.
"""

import json
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase, logging

from generative_data_prep.tokenized_line import (TokenizedArticle,
                                                 TokenizedSequence)
from generative_data_prep.utils import (BoundaryType, FileExtension,
                                        PackingConfig, TokenTypeIds)

from .sequence_packer import SequencePacker

DEFAULT_PACKING_CONFIG = PackingConfig.get_default()


class ArticleTokenizer:
    """Tokenize and pack text into sequences used for training NLP models."""

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_length: int,
                 file_ext: FileExtension,
                 packing_config: PackingConfig = DEFAULT_PACKING_CONFIG,
                 packing_boundary: BoundaryType = BoundaryType.JSONL,
                 attention_boundary: BoundaryType = BoundaryType.JSONL,
                 disable_space_separator: bool = False,
                 prompt_keyword: str = 'prompt',
                 completion_keyword: str = 'completion'):
        """Create Article Tokenizer.

        Args:
            tokenizer: Huggingface tokenizer that inherits from PreTrainedTokenizerBase and contains an
                encode function.
            max_seq_length: Maximum sequence length of model.
            file_ext: The file extension of input, whether to load .jsonl of .txt lines
            packing_config: How to pack tokens into sequences. Refer to SequencePacker class.
                Defaults to PackingStyleType.FULL.
            packing_boundary: Defines whether entire jsonl or prompt completion pair should be treated as
                one unit when running packing.Defaults to BoundaryType.JSONL.
            attention_boundary: How to define the boundary when attending to other tokens
            disable_space_separator: If true, do NOT add spaces between prompt completion pairs.
                Defaults to False.
            prompt_keyword: Keyword to index into loaded json dictionaries to get the prompting text.
                Defaults to 'prompt'.
            completion_keyword: Keyword to index into loaded json dictionaries to get the completion text.
                Defaults to 'completion'.

        Example:
            >>> input_text = [
            ...      '{"prompt": "hello how are", "completion": "you"}',
            ...      '{"prompt": "hi", "completion": "bye"}',
            ...      None
            ... ]
            >>> from transformers import GPT2Tokenizer
            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> article_tokenizer = ArticleTokenizer(tokenizer, 3, '.jsonl')
            >>> sequences = []
            >>> for text in input_text:
            ...     sequences += article_tokenizer(text)
            ...
            >>> sequences_str = '\\n'.join(map(str, sequences))
            >>> print(sequences_str)
            [(31373, <TokenTypeIds.PROMPT: 0>) (703, <TokenTypeIds.PROMPT: 0>) (389, <TokenTypeIds.PROMPT: 0>)]
            [(345, <TokenTypeIds.COMPLETION: 1>) (50256, <TokenTypeIds.SEP: 3>) (5303, <TokenTypeIds.PROMPT: 0>)]
            [(33847, <TokenTypeIds.COMPLETION: 1>) (50256, <TokenTypeIds.SEP: 3>) (50256, <TokenTypeIds.PADDING: 2>)]
        """
        self.tokenizer = tokenizer
        self.file_ext = file_ext
        self.packing_boundary = packing_boundary
        self.attention_boundary = attention_boundary
        self.disable_space_separator = disable_space_separator
        self.prompt_keyword = prompt_keyword
        self.completion_keyword = completion_keyword
        self.eos_token_id = tokenizer.eos_token_id
        self.packer = SequencePacker(max_seq_length, self.eos_token_id,
                                     packing_config)
        logging.set_verbosity_error()

    def __call__(self, article: Optional[str]) -> List[TokenizedSequence]:
        """Tokenize and pack input text into tokenized sequence.

        Takes an input line of text, tokenize it and then packs it into tokenized
        sequences. After packing this text using self.packer, this function returns
        any completed sequences, or an empty list. What defines a sequence as completed
        is the packing style, refer to SequencePacker for more information. If None
        is passed in then any remaining sequence that has not been completed are
        padded and returned.

        Args:
            article: Input text, or None

        Returns:
            List of tokenized sequences that have been completed
        """
        if article is None:
            return self.packer(article)

        tokenized_articles = []
        if self.file_ext == FileExtension.JSONL:
            # Load from json
            loaded_jsonl = json.loads(article)
            tokenized_articles += self.process_jsonl(loaded_jsonl)
        elif self.file_ext == FileExtension.TXT:
            # Load from txt
            tokenized_articles += self.process_text(article)
        else:
            err_msg = f"Input file extension {self.file_ext} is invalid,"
            err_msg += f" must be {FileExtension.JSONL} or {FileExtension.TXT}"
            raise ValueError(err_msg)

        tokenized_sequences = self.packer(tokenized_articles)

        return tokenized_sequences

    def process_text(self, text_line: str) -> List[TokenizedArticle]:
        """Take an input string, tokenize it, and return the tokenized article representation.

        Assumes that text is all "completion" tokens (it will be back-propogated on during training).

        Args:
            text_line: Input text line to use for training. Text within this line
            should be related and from the same context
        Returns:
            List with one element, that is the tokenized article representing the text_line
        """
        token_ids, token_type_ids = self.tokenize(text_line)

        if len(token_type_ids) >= 1:
            token_type_ids[-1] = TokenTypeIds.SEP

        tokenized_article = TokenizedArticle(token_ids, token_type_ids)
        return [tokenized_article]

    def process_jsonl(self, jsonl: Union[dict,
                                         List]) -> List[TokenizedArticle]:
        """Tokenize a loaded jsonl and store in a TokenizedArticle object.

        Takes in a loaded jsonl, and returns a List of tokenized articles based on self.BoundaryType.
        If self.packing_boundary is BoundaryType.PROMPT_COMPLETION_PAIR, then each tokenized article
        will contain a single tokenized prompt completion pair from the original jsonl.
        If self.packing_boundary is BoundaryType.JSONL, then each tokenized article will contain
        the tokenized text from all the prompt completion pairs in the jsonl

        Args:
            jsonl: The loaded input jsonl, in the form of {"prompt":"...", "completion":"..."}
            or [{"prompt":"...", "completion":"..."}, {"prompt":"...", "completion":"..."}, ...]
        Returns:
            Tokenized articles that represent input jsonl
        """
        if isinstance(jsonl, dict):
            jsonl = [jsonl]

        tokenized_articles = []
        token_ids, token_type_ids = [], []
        for i, prompt_completion in enumerate(jsonl):
            prompt = prompt_completion[
                self.
                prompt_keyword] if self.prompt_keyword in prompt_completion else ''
            if self.completion_keyword not in prompt_completion:
                err_msg = f'Completion keyword required in every jsonl, {self.completion_keyword} not found'
                raise json.JSONDecodeError(err_msg, str(jsonl), 0)
            completion = prompt_completion[self.completion_keyword]

            completion, prompt = self._add_space_separator(completion, prompt)
            new_token_ids, new_token_type_ids = self.tokenize(
                completion, prompt)
            token_ids += new_token_ids
            token_type_ids += new_token_type_ids

            if self.attention_boundary == BoundaryType.PROMPT_COMPLETION_PAIR and len(
                    token_type_ids) > 0:
                token_type_ids[-1] = TokenTypeIds.SEP
            if self.packing_boundary == BoundaryType.PROMPT_COMPLETION_PAIR and i != len(
                    jsonl) - 1:
                tokenized_article = TokenizedArticle(token_ids, token_type_ids)
                tokenized_articles.append(tokenized_article)
                token_ids, token_type_ids = [], []

        if len(token_type_ids) > 0:
            token_type_ids[-1] = TokenTypeIds.SEP
        tokenized_articles.append(TokenizedArticle(token_ids, token_type_ids))

        return tokenized_articles

    def _add_space_separator(self, completion: str,
                             prompt: str) -> Tuple[str, str]:
        """Remove any spaces between the prompt and completion and add a space before the completion.

        Args:
            completion: completion text
            prompt: prompt text

        Returns:
            completion, prompt with one space before the completion
        """
        if not self.disable_space_separator:
            if prompt:
                prompt = prompt.rstrip(' ')
            if completion:
                completion = ' ' + completion.lstrip(' ')

        return completion, prompt

    def tokenize(self,
                 completion: str,
                 prompt: Optional[str] = None) -> Tuple[List[int], List[int]]:
        """Tokenize the input prompt and completion.

        Call self.tokenizer.encode to convert the input prompt and completion into token ids.
        Creates token_type_id metadata for the tokens, and adds a self.tokenizer.eos_token_id to the end.

        Args:
            completion: completion text
            prompt: prompt text. Defaults to None.

        Returns:
            token_ids that represent input prompt and completion, and token_type_ids where
            each element represents the type (prompt, completion, eos, padding) of the
            token at the corresponding index in token_ids
        """
        token_ids: List[int] = []
        token_type_ids: List[int] = []

        if prompt:
            token_ids += self.tokenizer.encode(prompt)
            token_type_ids += len(token_ids) * [TokenTypeIds.PROMPT]

        if completion:
            completion_token_ids = self.tokenizer.encode(completion)
            token_ids += completion_token_ids
            token_type_ids += len(completion_token_ids) * [
                TokenTypeIds.COMPLETION
            ]

        if len(token_ids) >= 1:
            token_ids.append(self.eos_token_id)
            token_type_ids.append(TokenTypeIds.COMPLETION)

        return token_ids, token_type_ids
