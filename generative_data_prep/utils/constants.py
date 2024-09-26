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
"""

from enum import Enum

from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

CATEGORY_JSON_KEY = "category"


class BaseEnum(Enum):
    """Contains additional utility methods for the custom Enums."""

    @classmethod
    def as_list(cls):
        """Return the enum in the form of a list."""
        return [member.value for member in cls]


class BoundaryType(str, BaseEnum):
    """Represent the two differnt kinds of article boundaries.

    PROMPT_COMPLETION_PAIR: means that each prompt completion pair is
        treated as an article boundary, even if it is in the same jsonl
    JSONL: means that only different jsonls are trated as different
        articles
    """

    PROMPT_COMPLETION_PAIR = "prompt_completion_pair"
    JSONL = "jsonl"


class OverflowType(str, BaseEnum):
    """Represents the various options for overflowing.

    DROP: means that you drop any data points that don't fit in seqeunce length
    TRUNCATE_LEFT: means that if the article does not fit in the sequence length,
        you cut off the article from the left (the prompt)
    TRUNCATE_RIGHT: means that if the article does not fit in the sequence length,
        you gut off the article from the right (the completion)
    """

    DROP = "drop"
    TRUNCATE_LEFT = "truncate_left"
    TRUNCATE_RIGHT = "truncate_right"


class PackingStyleType(str, BaseEnum):
    """How to pack various articles into a sequence.

    GREEDY: Continue placing articles in the previos sequence until one does not fit
    FULL: place as much of the article as possible in the sequence, if it does not fit
        then just place as much of it as possible in the sequence, and continue placing
        the rest of the text in the next sequence
    SINGLE: Only place one article in each sequence
    """

    GREEDY = "greedy"
    FULL = "full"
    SINGLE = "single"


class TokenTypeIds(int, BaseEnum):
    """Metadata that keeps track of what kind of token each token is.

    PROMPT: tokens from the 'prompt' field of input jsonl
    COMPLETION: tokens from the 'completion' field of the input jsonl
    PADDING: tokens that pad sequences until they are full
    SEP: tokens that represent article boundaries, these tokens are always
        <|endoftext|> tokens, and draw the boundary between what tokens
        are attended to during training
    """

    PROMPT = 0
    COMPLETION = 1
    PADDING = 2
    SEP = 3


class FileExtension(str, BaseEnum):
    """Valid input file extensions."""

    TXT = ".txt"
    JSONL = ".jsonl"


class TokenizerConfigPair:
    """New datastructure to store tokenizer and config pairs for same model type."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: PretrainedConfig) -> None:
        """Initialization of TokenizerConfigPair.

        Args:
            tokenizer: Tokenizer associated with key
            config: Config associated with key
        """
        self.tokenizer = tokenizer
        self.config = config


GPT2_KEY = "gpt2"

TOKENIZER_CLASSES = {GPT2_KEY: TokenizerConfigPair(tokenizer=GPT2Tokenizer, config=GPT2Config)}
