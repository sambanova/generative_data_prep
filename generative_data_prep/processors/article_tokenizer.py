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


This module implements ArticleTokenizer.

Article tokenizer takes any input text or jsonl, runs tokenization using
the passed in tokenizer, and then packs the tokens into sequences using
the SequencePacker class.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase
from transformers import logging as transformers_logging

from generative_data_prep.tokenized_line import (
    Token,
    TokenizedArticle,
    TokenizedSequence,
)
from generative_data_prep.utils import (
    CATEGORY_JSON_KEY,
    BoundaryType,
    FileExtension,
    PackingConfig,
    TokenTypeIds,
)

from .metrics import Metrics
from .sequence_packer import SequencePacker

DEFAULT_PACKING_CONFIG = PackingConfig.get_default()
DEFAULT_PROMPT_PLACEHOLDER = "<prompt_placeholder>"
DEFAULT_COMPLETION_PLACEHOLDER = "<completion_placeholder>"


class ArticleTokenizer:
    """Tokenize and pack text into sequences used for training NLP models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        file_ext: FileExtension,
        packing_config: PackingConfig = DEFAULT_PACKING_CONFIG,
        packing_boundary: BoundaryType = BoundaryType.JSONL,
        attention_boundary: BoundaryType = BoundaryType.JSONL,
        disable_space_separator: bool = False,
        keep_prompt_only_sequences: bool = False,
        prompt_keyword: str = "prompt",
        completion_keyword: str = "completion",
        category_to_id: Optional[Dict[str, int]] = None,
        prompt_prefix: Optional[str] = None,
        prompt_postfix: Optional[str] = None,
        apply_chat_template: Optional[bool] = False,
    ):
        r"""Create Article Tokenizer.

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
            keep_prompt_only_sequences: If true, do NOT remove sequences that do not contain COMPLETION tokens
                Defaults to False.
            prompt_keyword: Keyword to index into loaded json dictionaries to get the prompting text.
                Defaults to 'prompt'.
            completion_keyword: Keyword to index into loaded json dictionaries to get the completion text.
                Defaults to 'completion'.
            category_to_id: Dictionary that maps category string names to IDs.
            prompt_prefix: text to add before the prompt, for chatML conventions use.
            prompt_postfix: text to add before the prompt, for chatML conventions use.
            apply_chat_template: If true, apply the chat template to the input jsonl before tokenizing.

        Example:
            >>> input_text = [
            ...      '{"prompt": "hello how", "completion": "are you"}',
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
            [(31373, <TokenTypeIds.PROMPT: 0>) (703, <TokenTypeIds.PROMPT: 0>) (389, <TokenTypeIds.COMPLETION: 1>)]
            [(345, <TokenTypeIds.COMPLETION: 1>) (50256, <TokenTypeIds.SEP: 3>) (5303, <TokenTypeIds.PROMPT: 0>)]
            [(33847, <TokenTypeIds.COMPLETION: 1>) (50256, <TokenTypeIds.SEP: 3>) (50256, <TokenTypeIds.PADDING: 2>)]
        """
        self.tokenizer = tokenizer
        self.file_ext = file_ext
        self.packing_boundary = packing_boundary
        self.attention_boundary = attention_boundary
        self.disable_space_separator = disable_space_separator
        self.keep_prompt_only_sequences = keep_prompt_only_sequences
        self.prompt_keyword = prompt_keyword
        self.completion_keyword = completion_keyword
        self.eos_token_id = tokenizer.eos_token_id
        self.metrics = Metrics()
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        self.packer = SequencePacker(max_seq_length, pad_token_id, packing_config, self.metrics)
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        transformers_logging.set_verbosity_error()

        self.logged_prompt_only_warn_msg_prepack = False
        self.category_to_id = category_to_id

        self.apply_chat_template = apply_chat_template

    def _update_token_metrics(self, tokenized_sequences: List[TokenizedSequence]):
        """Update the token metrics for the finished tokenized sequences that have been packed.

        Args:
            tokenized_sequences: The tokenized sequences that are finished being packed.
        """
        self.metrics.sequences += len(tokenized_sequences)
        for seq in tokenized_sequences:
            self.metrics.output_tokens += len(seq)
            self.metrics.prompt_tokens += seq.prompt_tokens
            self.metrics.completion_tokens += seq.completion_tokens
            self.metrics.padding_tokens += seq.pad_tokens

    def _convert_jsonl_to_user_assitant(self, jsonl: Union[dict, List]) -> Tuple[List[dict], List[str], List[str]]:
        """Convert the input jsonl into 'user', 'assistant' format. Put placeholder prompt and completions.

        Convert the jsonl to 'user', 'assistant' format so chat tempalte can be applied.
        Place temporary <prompt_placeholder> and <completion_placeholder> fags in the string,
        and save the actual prompts and completions in a seperate list to fill in later.

        Args:
            jsonl: The input data from input file

        Returns:
            Tuple[List[dict], List[str], List[str]]: In order, this returns the user assistant list,
            then the prompt list and finally the completion list.
        """
        converted_jsonl = []
        prompts = []
        completions = []

        for prompt_completion in jsonl:
            if self.completion_keyword not in prompt_completion:
                err_msg = f"Completion keyword required in every jsonl, {self.completion_keyword} not found"
                raise json.JSONDecodeError(err_msg, str(jsonl), 0)

            prompt = prompt_completion[self.prompt_keyword] if self.prompt_keyword in prompt_completion else ""
            completion = prompt_completion[self.completion_keyword]
            converted_jsonl.append({"role": "user", "content": DEFAULT_PROMPT_PLACEHOLDER})
            converted_jsonl.append({"role": "assistant", "content": DEFAULT_COMPLETION_PLACEHOLDER})

            prompts.append(prompt)
            completions.append(completion)

        if len(prompts) != len(completions):
            err_msg = f"Number of prompts and completions must be equal, \
             {len(prompts)} prompts and {len(completions)} completions found"
            raise ValueError(err_msg)

        if len(prompts) != len(jsonl):
            err_msg = f"Number of prompts and jsonl must be equal, {len(prompts)} prompts and {len(jsonl)} jsonl found"
            raise ValueError(err_msg)

        return converted_jsonl, prompts, completions

    def _tokenize_chat_templated_data(
        self, split_chat: List[str], prompt_prefix: str, prompts: List[str], completions: List[str]
    ) -> List[TokenizedArticle]:
        """Iterate through split_chat and create prompt completion pairs and return tokenized text.

        Args:
            split_chat: The list of text to tokenier split by prompts and completions
            prompt_prefix: The text occuring right before every prompt
            prompts: List of the prompts
            completions: list of completions
        """
        tokenized_articles = []
        tokens = []
        # If we are currently iterating over prompt text or completion text
        state = "PROMPT"
        curr_prompt_text = ""
        prompts_index = 0
        curr_completion_text = ""
        completions_index = 0

        for split in split_chat:
            if split == DEFAULT_PROMPT_PLACEHOLDER:
                curr_prompt_text += prompts[prompts_index]
                prompts_index += 1
            elif split == DEFAULT_COMPLETION_PLACEHOLDER:
                curr_completion_text += completions[completions_index]
                completions_index += 1
                state = "COMPLETION"
            else:
                # If we are in the prompt state just add to the prompt
                if state == "PROMPT":
                    curr_prompt_text += split
                elif state == "COMPLETION":
                    # If the text ends with prompt prefix, then we should switch to the next prompt
                    if split.endswith(prompt_prefix):
                        # We have finished the current article, so tokenize the text and add it
                        curr_completion_text += split.split(prompt_prefix)[0]
                        tokens += self.tokenize(
                            completion=curr_completion_text,
                            prompt=curr_prompt_text,
                            apply_chat_template=True,
                        )

                        if self.attention_boundary == BoundaryType.PROMPT_COMPLETION_PAIR and len(tokens) > 0:
                            tokens[-1].make_article_boundary()
                        if (
                            self.packing_boundary == BoundaryType.PROMPT_COMPLETION_PAIR
                            and prompts_index != len(split_chat) - 1
                        ):
                            tokenized_article = TokenizedArticle(tokens)
                            tokenized_articles.append(tokenized_article)
                            tokens = []

                        curr_prompt_text = prompt_prefix
                        curr_completion_text = ""
                        state = "PROMPT"
                    else:
                        curr_completion_text += split

        if len(tokens) > 0:
            tokens[-1].make_article_boundary()
        tokenized_articles.append(TokenizedArticle(tokens))

        return tokenized_articles

    def process_jsonl_chat(self, jsonl: Union[dict, List]) -> List[TokenizedArticle]:
        """Tokenize a loaded jsonl with the chat template applied and return TokenizedArticle object.

        Because we need to keep the metadata about what token is prompt, what token is completion and what is EOS
        The following steps are performed:
        1. Convert the jsonl into "user" "assistant" chat format
        2. Replace the prompt and completion with <prompt_placeholder> and <completion_placeholder>
        3. Apply the chat template
        5. Replace the <prompt_placeholder> and <completion_placeholder> tags with the actual prompt and completion
        5. Tokenize the newly constructed prompt completion pairs

        Args:
            jsonl (Union[dict, List]): A list of dictionaries where each dictionary represents a JSONL data item.
            Each dictionary should contain 'prompt' and 'completion' keys.
            The loaded input jsonl, in the form of {"prompt":"...", "completion":"..."}
            or [{"prompt":"...", "completion":"..."}, {"prompt":"...", "completion":"..."}, ...]

        Returns:
            List[TokenizedArticle]: Tokenized articles representation of the input jsonl, with chat template.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            err_msg = (
                "--apply_chat_template used, but the tokenizer does not have 'apply_chat_template' method defined."
            )
            raise ValueError(err_msg)

        # If jsonl is a dict, make it a single element list to guarantee type of List[Dict]
        if isinstance(jsonl, dict):
            jsonl = [jsonl]

        # convert jsonl from prompt completion format to user assistant chat format
        converted_jsonl, prompts, completions = self._convert_jsonl_to_user_assitant(jsonl)

        # Apply specified chat templates using tokenizer (apply_chat_template) without tokenization.
        formatted_chat = self.tokenizer.apply_chat_template(converted_jsonl, tokenize=False)

        # Split by the prompt placeholder and completion placeholder
        pattern = rf"({DEFAULT_PROMPT_PLACEHOLDER}|{DEFAULT_COMPLETION_PLACEHOLDER})"
        split_chat = re.split(pattern, formatted_chat)

        # Search for the start instruction of each conversation.
        prompt_prefix = ""
        for tok in split_chat:
            if tok == DEFAULT_PROMPT_PLACEHOLDER:
                break
            prompt_prefix += tok

        # Construct prompt completion pairs from the split chat and tokenize them
        tokenized_articles = self._tokenize_chat_templated_data(split_chat, prompt_prefix, prompts, completions)

        return tokenized_articles

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
            tokenized_sequences = self.packer(article)
            if not self.keep_prompt_only_sequences:
                tokenized_sequences = self._remove_prompt_only_sequences(tokenized_sequences)
            self._update_token_metrics(tokenized_sequences)
            return tokenized_sequences

        self.metrics.articles += 1
        tokenized_articles = []
        if self.file_ext == FileExtension.JSONL:
            # Load from json
            loaded_jsonl = json.loads(article)
            if self.apply_chat_template:
                tokenized_articles += self.process_jsonl_chat(loaded_jsonl)
            else:
                tokenized_articles += self.process_jsonl(loaded_jsonl)
        elif self.file_ext == FileExtension.TXT:
            # Load from txt
            tokenized_articles += self.process_text(article)
        else:
            err_msg = f"Input file extension {self.file_ext} is invalid,"
            err_msg += f" must be {FileExtension.JSONL} or {FileExtension.TXT}"
            raise ValueError(err_msg)

        tokenized_sequences = self.packer(tokenized_articles)

        # Check if any sequence in tokenized_sequences contains no completion tokens
        if not self.keep_prompt_only_sequences:
            tokenized_sequences = self._remove_prompt_only_sequences(tokenized_sequences)

        self._update_token_metrics(tokenized_sequences)

        return tokenized_sequences

    def _remove_prompt_only_sequences(self, tokenized_sequences: List[TokenizedSequence]) -> List[TokenizedSequence]:
        """Takes a list of TokenizedSequences, removes those that don't contain any COMPLETION TokenTypeIds.

        Args:
            tokenized_sequences: List of TokenizedSequence
        Returns:
            Original list with prompt-only sequences filtered out
        """
        filtered_sequences = []

        for seq in tokenized_sequences:
            if TokenTypeIds.COMPLETION not in seq.dump_token_type_ids():
                self.metrics.tokens_dropped_from_all_prompt += len(seq)
                continue
            filtered_sequences.append(seq)

        return filtered_sequences

    def process_text(self, text_line: str) -> List[TokenizedArticle]:
        """Take an input string, tokenize it, and return the tokenized article representation.

        Assumes that text is all "completion" tokens (it will be back-propogated on during training).

        Args:
            text_line: Input text line to use for training. Text within this line
            should be related and from the same context
        Returns:
            List with one element, that is the tokenized article representing the text_line
        """
        tokens = self.tokenize(text_line)

        if len(tokens) >= 1:
            tokens[-1].make_article_boundary()

        tokenized_article = TokenizedArticle(tokens)
        return [tokenized_article]

    def get_category_id(self, prompt_completion):
        """Extract the category id metadata from the category_id key of the loaded jsonl.

        Args:
            prompt_completion: The loaded jsonl.

        Returns:
            The category ID of the jsonl, save this metadata in HDF5 files.
        """
        category_id = -1
        if self.category_to_id is not None and CATEGORY_JSON_KEY in prompt_completion:
            category_name = prompt_completion[CATEGORY_JSON_KEY]
            if category_name not in self.category_to_id:
                err = f"jsonl found with key {CATEGORY_JSON_KEY} and value {category_name},"
                err += f" but this category name is not in inputted --categories_path flag {self.category_to_id}"
                raise ValueError(err)
            category_id = self.category_to_id[category_name]

        return category_id

    def process_jsonl(self, jsonl: Union[dict, List]) -> List[TokenizedArticle]:
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
        tokens = []
        for i, prompt_completion in enumerate(jsonl):
            prompt = prompt_completion[self.prompt_keyword] if self.prompt_keyword in prompt_completion else ""
            if self.completion_keyword not in prompt_completion:
                err_msg = f"Completion keyword required in every jsonl, {self.completion_keyword} not found"
                raise json.JSONDecodeError(err_msg, str(jsonl), 0)
            completion = prompt_completion[self.completion_keyword]

            if not completion and not self.keep_prompt_only_sequences:
                continue

            category_id = self.get_category_id(prompt_completion)
            completion, prompt = self._add_space_separator(completion, prompt)
            tokens += self.tokenize(completion, prompt, category_id)

            if self.attention_boundary == BoundaryType.PROMPT_COMPLETION_PAIR and len(tokens) > 0:
                tokens[-1].make_article_boundary()
            if self.packing_boundary == BoundaryType.PROMPT_COMPLETION_PAIR and i != len(jsonl) - 1:
                tokenized_article = TokenizedArticle(tokens)
                tokenized_articles.append(tokenized_article)
                tokens = []

        if len(tokens) > 0:
            tokens[-1].make_article_boundary()
        tokenized_articles.append(TokenizedArticle(tokens))

        return tokenized_articles

    def _add_space_separator(self, completion: str, prompt: str) -> Tuple[str, str]:
        """Remove any spaces between the prompt and completion and add a space before the completion.

        Args:
            completion: completion text
            prompt: prompt text

        Returns:
            completion, prompt with one space before the completion
        """
        if not self.disable_space_separator:
            if prompt:
                prompt = prompt.rstrip(" ")
            if completion:
                completion = " " + completion.lstrip(" ")

        return completion, prompt

    def tokenize(
        self,
        completion: str,
        prompt: Optional[str] = None,
        category_id: Optional[int] = -1,
        apply_chat_template: bool = False,
    ) -> List[Token]:
        """Tokenize the input prompt and completion.

        Call self.tokenizer.encode to convert the input prompt and completion into token ids.
        Creates token_type_id metadata for the tokens, and adds a self.tokenizer.eos_token_id to the end.

        Args:
            completion: completion text
            prompt: prompt text. Defaults to None.
            category_id: The category to assign these tokens to
            apply_chat_template: Whether a chat template will be applied to these tokens

        Returns:
            token_ids that represent input prompt and completion, and token_type_ids where
            each element represents the type (prompt, completion, eos, padding) of the
            token at the corresponding index in token_ids
        """
        tokens = []
        if prompt:
            if self.prompt_prefix:
                prompt = self.prompt_prefix + prompt
            if self.prompt_postfix:
                prompt = prompt + self.prompt_postfix
            prompt_token_ids = self.tokenizer.encode(prompt)
            # If there is an EOS token at the end of the prompt then remove it
            if (
                hasattr(self.tokenizer, "eos_token_id")
                and prompt_token_ids[-1] == self.tokenizer.eos_token_id
                and len(prompt_token_ids) >= 1
            ):
                prompt_token_ids = prompt_token_ids[:-1]
            # If there is a BOS token at the begining of the prompt, and apply chat template
            # is used, then remove the extra BOS
            if (
                hasattr(self.tokenizer, "bos_token_id")
                and len(prompt_token_ids) >= 1
                and prompt_token_ids[0] == self.tokenizer.bos_token_id
                and apply_chat_template
            ):
                prompt_token_ids = prompt_token_ids[1:]
            tokens += list(map(lambda x: Token(x, TokenTypeIds.PROMPT, category_id), prompt_token_ids))

        if completion:
            completion_token_ids = self.tokenizer.encode(completion)
            # If there is a BOS token at the begining of the completion then remove it
            # Only remove it if there are prompt tokens
            if (
                hasattr(self.tokenizer, "bos_token_id")
                and completion_token_ids[0] == self.tokenizer.bos_token_id
                and (prompt and len(prompt_token_ids) >= 0)
                and len(completion_token_ids) >= 1
            ):
                completion_token_ids = completion_token_ids[1:]
            # If there is an EOS token at the end of completion then remove it
            if (
                hasattr(self.tokenizer, "eos_token_id")
                and completion_token_ids[-1] == self.tokenizer.eos_token_id
                and len(completion_token_ids) >= 1
            ):
                completion_token_ids = completion_token_ids[:-1]
            tokens += list(map(lambda x: Token(x, TokenTypeIds.COMPLETION, category_id), completion_token_ids))

        if len(tokens) >= 1:
            tokens.append(Token(self.eos_token_id, TokenTypeIds.COMPLETION, category_id))

        self.metrics.input_tokens += len(tokens)
        return tokens
