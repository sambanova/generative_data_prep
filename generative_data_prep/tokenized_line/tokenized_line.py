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


This module implements data structures to represent text that has been tokenized.

All data structures inherit from the base class TokenizedLine.   TokenizedArticle represents newly tokenized text.
TokenizedSequence represents TokenizedArticles that have been further processed and are ready to be trained on by
our models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Union, overload

from generative_data_prep.utils import TokenTypeIds

from .token import Token

# custom type representing subclasses of TokenizedLine
TokenizedLineSubClass = TypeVar("TokenizedLineSubClass", bound="TokenizedLine")


class TokenizedLine(ABC):
    """Represent a line of text that has been tokenized into token ids and token type ids."""

    def __init__(self, tokens: List[Token]):
        """Create a TokenizedLine.

        Args:
            tokens: list of tokens that make up this tokenized line. These tokens have two main attributes
                token_ids:  The token ids corresponding to the tokens that the line was tokenized into.  For example,
                    if the line was "how are you doing", the tokens might be ["how", "are", "you", "do", "ing"],
                    and the corresponding token_ids might be [293, 43, 3992, 499, 32].  The actual ids are subject to
                    the tokenizer's implementation.
                token_type_ids:  The token type ids corresponding to the token ids.  The token type ids specify the type
                    of each corresponding token.  Some examples of token type ids include 'Prompt', 'Completion, and
                    'End of Sequence'.
        """
        self._tokens = tokens

    def __iadd__(self: TokenizedLineSubClass, tokenized_line: "TokenizedLine") -> TokenizedLineSubClass:
        """Implement += for a tokenized line."""
        self._tokens += tokenized_line.tokens

        return self

    @overload
    def __getitem__(self: TokenizedLineSubClass, index: slice) -> TokenizedLineSubClass:
        """See __getitem__ docstring below, this is just a type hint."""  # noqa: D418
        ...

    @overload
    def __getitem__(self: TokenizedLineSubClass, index: int) -> Token:
        """See __getitem__ docstring below, this is just a type hint."""  # noqa: D418
        ...

    def __getitem__(self: TokenizedLineSubClass, index: Union[int, slice]) -> Union[Token, TokenizedLineSubClass]:
        """Return the token ID and the token type ID at the specified index / slice."""
        if isinstance(index, slice):
            return self._get_slice(index)
        elif isinstance(index, int):
            return self.tokens[index]
        else:
            raise TypeError(f"Invalid type: {type(index)}")

    def __len__(self) -> int:
        """Return the length of the tokenized line."""
        return len(self.tokens)

    def __str__(self) -> str:
        """Return the tokenized line as a string."""
        return f'[{" ".join(map(str, self.tokens))}]'

    def __repr__(self) -> str:
        """Return the tokenized line representation.

        Currently the string representation of the tokenized line uniquely identifies a tokenized line, so we just
        call the string function here.
        """
        return str(self)

    def __eq__(self, obj: object) -> bool:
        """Return whether or not another TokenizedLine is equal to this one."""
        if not isinstance(obj, TokenizedLine):
            return False
        return self.tokens == obj.tokens

    @property
    def tokens(self) -> List[Token]:
        """Return the token ids of the TokenizedLine."""
        return self._tokens

    def is_empty(self) -> bool:
        """Return whether or not the TokenizedLine is empty."""
        return len(self) == 0

    @abstractmethod
    def _get_slice(self: TokenizedLineSubClass, slice_index: slice) -> TokenizedLineSubClass:
        """Return a slice of the TokenizedLine.

        Args:
            slice_index: A slice (which contains a start and end index).
        """
        raise NotImplementedError

    def dump_token_ids(self):
        """Return a list of the token ids from this tokenixed line."""
        return list(map(lambda x: x.token_id, self.tokens))

    def dump_token_type_ids(self):
        """Return a list of the token type ids from this tokenixed line."""
        return list(map(lambda x: x.token_type_id, self.tokens))

    def dump_category_ids(self):
        """Return a list of the token type ids from this tokenixed line."""
        return list(map(lambda x: x.category_id, self.tokens))


class TokenizedArticle(TokenizedLine):
    """Represents an article that has been tokenized.

    An article is a block of semantically related text like a paragraph, or a conversation, etc. depending on the
    textual data you are training on.
    """

    @classmethod
    def get_empty(cls) -> "TokenizedArticle":
        """See base class."""
        return cls([])

    def _get_slice(self, slice_index: slice) -> "TokenizedArticle":
        """See base class."""
        return TokenizedArticle(self.tokens[slice_index])


class TokenizedSequence(TokenizedLine):
    """Represents a sequence of tokens that can be ingested by our models for training.

    TokenizedSequences are essentially Tokenized Articles but in addition have a maximum length.  Transformer models
    require the training sequence of tokens to be bounded by a certain length, which is why we can't directly use
    the TokenizedArticles, and must first compress the TokenizedArticles into length-bounded TokenizedSequences.
    """

    def __init__(self, tokens: List[Token], max_seq_length: int, pad_token_id: Optional[int] = None):
        """Create a TokenizedSequence.

        Args:
            tokens:  The tokens that make up this tokenized line, each token has an id, a type_id
            max_seq_length:  The maximum allowed length for a sequence.
            pad_token_id: If this tokenizer has a unique padding token id, use this.
        """
        if max_seq_length < 1:
            err_msg = f"Cannot have zero / negative max_seq_length. Found max_seq_length == {max_seq_length}"
            raise ValueError(err_msg)
        if len(tokens) > max_seq_length:
            err_msg = f"Tokens have length == {len(tokens)}, expected length to be <= {max_seq_length}"
            raise ValueError(err_msg)
        super().__init__(tokens)
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

    @classmethod
    def get_empty(cls, max_seq_length: int, pad_token_id: int) -> "TokenizedSequence":
        """See base class."""
        return cls.from_article(TokenizedArticle.get_empty(), max_seq_length, pad_token_id)  # type: ignore

    @classmethod
    def from_article(cls, tokenized_article: TokenizedArticle, max_seq_length: int, pad_token_id: int):
        """Create a TokenizedLine from a TokenizedArticle.

        Args:
            tokenized_article: The tokenized article.
            max_seq_length:  The maximum sequence length for this new TokenizedSequence.
            pad_token_id: The token used to pad sequence up to sequence length.

        Returns:
            The newly created TokenizedLine.
        """
        return cls(tokenized_article.tokens, max_seq_length, pad_token_id)

    def __iadd__(self, tokenized_line: TokenizedLine) -> "TokenizedSequence":
        """Add another TokenizedLine to this TokenizedSequence.

        The token IDs and token type IDs of the TokenizedLine will be concatenated to the token IDs and token type IDs
        of this TokenizedSequence.

        Args:
            tokenized_line:  The TokenizedLine to be added.

        Returns:
            The resulting TokenizedSequence.
        """
        if len(self) + len(tokenized_line) > self.max_seq_length:
            err_msg_1 = f"Tokenized line with length: {len(tokenized_line)} is too long to be added to"
            err_msg_2 = f"sequence with length: {len(self)} and max sequence length: {self.max_seq_length}"
            raise ValueError(f"{err_msg_1} {err_msg_2}")
        return super().__iadd__(tokenized_line)

    @property
    def free_tokens(self):
        """The number of unfilled tokens in this sequence."""
        return self.max_seq_length - len(self)

    @property
    def prompt_tokens(self):
        """The number of prompt tokens in this sequence."""
        return sum(token.token_type_id == TokenTypeIds.PROMPT for token in self.tokens)

    @property
    def completion_tokens(self):
        """The number of completion tokens in this sequence."""
        return sum(token.token_type_id in [TokenTypeIds.COMPLETION, TokenTypeIds.SEP] for token in self.tokens)

    @property
    def pad_tokens(self):
        """The number of padding tokens in this sequence."""
        return sum(token.token_type_id == TokenTypeIds.PADDING for token in self.tokens)

    def is_packed(self) -> bool:
        """Return whether or not the TokenizedSequence is at its maximum length."""
        return len(self.tokens) == self.max_seq_length

    def pack(self, tokenized_line: TokenizedLineSubClass) -> TokenizedLineSubClass:
        """Pack a TokenizedLine into this TokenizedSequence.

        Add as much of a TokenizedLine as possible to this TokenizedSequence.

        Args:
            tokenized_line:  The TokenizedLine to be packed into this sequence.

        Returns:
            The left over portion of the TokenizedLine.
        """
        slice_index = self.max_seq_length - len(self)
        self += tokenized_line[:slice_index]
        return tokenized_line[slice_index:]

    def pad(self):
        """Fill the remaining token ids in the TokenizedSequence with padding tokens."""
        padding_size = self.max_seq_length - len(self)
        self._tokens += [Token(self.pad_token_id, TokenTypeIds.PADDING)] * padding_size

    def _get_slice(self, slice_index: slice) -> "TokenizedSequence":
        """See base class."""
        tokenized_article = TokenizedArticle(self.tokens[slice_index])
        return TokenizedSequence.from_article(tokenized_article, self.max_seq_length, self.pad_token_id)  # type: ignore
