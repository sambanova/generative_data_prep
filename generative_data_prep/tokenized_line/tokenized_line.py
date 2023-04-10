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


This module implements data structures to represent text that has been tokenized.

All data structures inherit from the base class TokenizedLine.   TokenizedArticle represents newly tokenized text.
TokenizedSequence represents TokenizedArticles that have been further processed and are ready to be trained on by
our models.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Union, overload

from generative_data_prep.utils import TokenTypeIds

# custom type representing subclasses of TokenizedLine
TokenizedLineSubClass = TypeVar('TokenizedLineSubClass', bound='TokenizedLine')


class TokenizedLine(ABC):
    """Represent a line of text that has been tokenized into token ids and token type ids."""

    def __init__(self, token_ids: List[int], token_type_ids: List[int]):
        """Create a TokenizedLine.

        Args:
            token_ids:  The token ids corresponding to the tokens that the line was tokenized into.  For example, if the
                line was "how are you doing", the tokens might be ["how", "are", "you", "do", "ing"], and the
                corresponding token_ids might be [293, 43, 3992, 499, 32].  The actual ids are subject to the
                tokenizer's implementation.
            token_type_ids:  The token type ids corresponding to the token ids.  The token type ids specify the type
                of each corresponding token.  Some examples of token type ids include 'Prompt', 'Completion, and 'End
                of Sequence'.
        """
        err_msg = f'Length of Token IDs {len(token_ids)} must match length of Token Type IDs {len(token_type_ids)}'
        assert len(token_ids) == len(token_type_ids), err_msg
        self._token_ids = token_ids
        self._token_type_ids = token_type_ids

    def __iadd__(self: TokenizedLineSubClass,
                 tokenized_line: 'TokenizedLine') -> TokenizedLineSubClass:
        """Implement += for a tokenized line."""
        self._token_type_ids += tokenized_line.token_type_ids
        self._token_ids += tokenized_line.token_ids
        return self

    @overload
    def __getitem__(self: TokenizedLineSubClass,
                    index: slice) -> TokenizedLineSubClass:
        """See __getitem__ docstring below, this is just a type hint."""
        ...

    @overload
    def __getitem__(self: TokenizedLineSubClass,
                    index: int) -> Tuple[int, int]:
        """See __getitem__ docstring below, this is just a type hint."""
        ...

    def __getitem__(
        self: TokenizedLineSubClass,
        index: Union[int,
                     slice]) -> Union[Tuple[int, int], TokenizedLineSubClass]:
        """Return the token ID and the token type ID at the specified index / slice."""
        if isinstance(index, slice):
            return self._get_slice(index)
        elif isinstance(index, int):
            return self.token_ids[index], self.token_type_ids[index]
        else:
            raise TypeError(f'Invalid type: {type(index)}')

    def __len__(self) -> int:
        """Return the length of the tokenized line."""
        return len(self.token_ids)

    def __str__(self) -> str:
        """Return the tokenized line as a string."""
        return f'[{" ".join(map(str, zip(self.token_ids, self.token_type_ids)))}]'

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
        return self.token_type_ids == obj.token_type_ids and self.token_ids == obj.token_ids

    @property
    def token_ids(self) -> List[int]:
        """Return the token ids of the TokenizedLine."""
        return self._token_ids

    @property
    def token_type_ids(self) -> List[int]:
        """Return the token type ids of the TokenizedLine."""
        return self._token_type_ids

    def is_empty(self) -> bool:
        """Return whether or not the TokenizedLine is empty."""
        return len(self) == 0

    @abstractmethod
    def _get_slice(self: TokenizedLineSubClass,
                   slice_index: slice) -> TokenizedLineSubClass:
        """Return a slice of the TokenizedLine.

        Args:
            slice_index: A slice (which contains a start and end index).
        """
        raise NotImplementedError


class TokenizedArticle(TokenizedLine):
    """Represents an article that has been tokenized.

    An article is a block of semantically related text like a paragraph, or a conversation, etc. depending on the
    textual data you are training on.
    """

    @classmethod
    def get_empty(cls) -> 'TokenizedArticle':
        """See base class."""
        return cls([], [])

    def _get_slice(self, slice_index: slice) -> 'TokenizedArticle':
        """See base class."""
        return TokenizedArticle(self.token_ids[slice_index],
                                self.token_type_ids[slice_index])


class TokenizedSequence(TokenizedLine):
    """Represents a sequence of tokens that can be ingested by our models for training.

    TokenizedSequences are essentially Tokenized Articles but in addition have a maximum length.  Transformer models
    require the training sequence of tokens to be bounded by a certain length, which is why we can't directly use
    the TokenizedArticles, and must first compress the TokenizedArticles into length-bounded TokenizedSequences.
    """

    def __init__(self, token_ids: List[int], token_type_ids: List[int],
                 max_seq_length: int, eos_token_id: int):
        """Create a TokenizedSequence.

        Args:
            token_ids:  The token ids (see TokenizedLine for more details).
            token_type_ids:  The token type ids (see TokenizedLine for more details).
            max_seq_length:  The maximum allowed length for a sequence.
            eos_token_id:  The end of text (sequence) token.  If a sequence's length is less than the max sequence
                length, the sequence is usually padded with this token.
        """
        err_msg = f'Cannot have zero / negative max_seq_length. Found max_seq_length == {max_seq_length}'
        assert max_seq_length >= 1, err_msg
        err_msg = f'Token IDs have length == {len(token_ids)}, expected length to be <= {max_seq_length}'
        assert len(token_ids) <= max_seq_length, err_msg
        super().__init__(token_ids, token_type_ids)
        self.max_seq_length = max_seq_length
        self.eos_token_id = eos_token_id

    @classmethod
    def get_empty(cls, max_seq_length: int,
                  eos_token_id: int) -> 'TokenizedSequence':
        """See base class."""
        return cls.from_article(TokenizedArticle.get_empty(), max_seq_length,
                                eos_token_id)

    @classmethod
    def from_article(cls, tokenized_article: TokenizedArticle,
                     max_seq_length: int, eos_token_id: int):
        """Create a TokenizedLine from a TokenizedArticle.

        Args:
            tokenized_article: The tokenized article.
            max_seq_length:  The maximum sequence length for this new TokenizedSequence.
            eos_token_id:  The end of text token for this new TokenizedSequence.
        Returns:
            The newly created TokenizedLine.
        """
        return cls(tokenized_article.token_ids,
                   tokenized_article.token_type_ids, max_seq_length,
                   eos_token_id)

    def __iadd__(self, tokenized_line: TokenizedLine) -> 'TokenizedSequence':
        """Add another TokenizedLine to this TokenizedSequence.

        The token IDs and token type IDs of the TokenizedLine will be concatenated to the token IDs and token type IDs
        of this TokenizedSequence.

        Args:
            tokenized_line:  The TokenizedLine to be added.
        Returns:
            The resulting TokenizedSequence.
        """
        err_msg_1 = f'Tokenized line with length: {len(tokenized_line)} is too long to be added to'
        err_msg_2 = f'sequence with length: {len(self)} and max sequence length: {self.max_seq_length}'
        assert len(self) + len(
            tokenized_line) <= self.max_seq_length, f'{err_msg_1} {err_msg_2}'
        return super().__iadd__(tokenized_line)

    @property
    def free_tokens(self):
        return self.max_seq_length - len(self)

    def is_packed(self) -> bool:
        """Return whether or not the TokenizedSequence is at its maximum length."""
        return len(self.token_ids) == self.max_seq_length

    def pack(self,
             tokenized_line: TokenizedLineSubClass) -> TokenizedLineSubClass:
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
        """Fill the remaining token ids in the TokenizedSequence with the end of text token."""
        padding_size = self.max_seq_length - len(self)
        self._token_type_ids += padding_size * [TokenTypeIds.PADDING]
        self._token_ids += padding_size * [self.eos_token_id]

    def _get_slice(self, slice_index: slice) -> 'TokenizedSequence':
        """See base class."""
        tokenized_article = TokenizedArticle(self.token_ids[slice_index],
                                             self.token_type_ids[slice_index])
        return TokenizedSequence.from_article(tokenized_article,
                                              self.max_seq_length,
                                              self.eos_token_id)
