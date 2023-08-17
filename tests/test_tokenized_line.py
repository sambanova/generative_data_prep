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
"""

from typing import List

import pytest

from generative_data_prep.tokenized_line import (
    TokenizedArticle,
    TokenizedLine,
    TokenizedSequence,
)
from generative_data_prep.utils import TokenTypeIds


@pytest.mark.parametrize("length,max_seq_length,eos_token_id", [(5, None, None), (2, 6, -1), (0, 6, -1)])
def test_len(tokenized_line: TokenizedLine, length: int):
    """Basic length test."""
    assert len(tokenized_line) == length
    assert len(tokenized_line.token_ids) == length
    assert len(tokenized_line.token_type_ids) == length


@pytest.mark.parametrize(
    "length,max_seq_length,eos_token_id,expected",
    [
        (3, None, None, "[(0, 0) (1, -1) (2, -2)]"),
    ],
)
def test_str(tokenized_line: TokenizedLine, expected: str):
    """String test."""
    assert str(tokenized_line) == repr(tokenized_line) == expected


@pytest.mark.fast
@pytest.mark.parametrize(
    "length,length_2,max_seq_length,max_seq_length_2,eos_token_id,eos_token_id_2",
    [(5, 4, None, None, None, None), (9, 2, 12, 12, -1, -1), (9, 1, None, 2, None, -1)],
)
def test_add(
    tokenized_line: TokenizedLine,
    tokenized_line_2: TokenizedLine,
    length: int,
    length_2: int,
):
    """Test that one tokenized line can be added to another tokenized line."""
    tokenized_line += tokenized_line_2
    assert len(tokenized_line) == length + length_2


@pytest.mark.parametrize(
    "length,length_2,max_seq_length,max_seq_length_2,eos_token_id,eos_token_id_2",
    [(1, 9, 2, None, -1, None)],
)
def test_illegal_add(tokenized_line: TokenizedLine, tokenized_line_2: TokenizedLine):
    """Test that the sequence's max sequence length is respected when adding another line to a sequence."""
    with pytest.raises(ValueError):
        tokenized_line += tokenized_line_2


@pytest.mark.parametrize("length,max_seq_length,eos_token_id,index", [(10, None, None, 4), (10, 12, -1, 0)])
def test_get_item(tokenized_line: TokenizedLine, index: int):
    """Test integer indexing of the tokenized line."""
    token, token_type_id = tokenized_line[index]
    # this is based on the way we initialized token_ids in the
    # tokenized_line fixture, in reality the token_ids and token type ids
    # could be anything
    assert token == index
    assert token_type_id == -index


@pytest.mark.fast
@pytest.mark.parametrize(
    "length,max_seq_length,eos_token_id,index_slice",
    [
        (10, None, None, slice(4, 6)),
        (10, 12, -1, slice(0, 9)),
        (10, 12, -1, slice(0, 9, 2)),
    ],
)
def test_get_slice(tokenized_line: TokenizedLine, index_slice: slice):
    """Test slice indexing of the tokenized line."""
    if index_slice.step is None:
        index_slice = slice(index_slice.start, index_slice.stop, 1)

    tokenized_line_slice = tokenized_line[index_slice]
    token_ids = tokenized_line_slice.token_ids
    token_type_ids = tokenized_line_slice.token_type_ids
    # as mentioned before, this is based on the way we initialized token_ids in the
    # tokenized_line fixture.
    for i, index in enumerate(range(index_slice.start, index_slice.stop, index_slice.step)):
        assert token_ids[i] == index
        assert token_type_ids[i] == -index


@pytest.mark.fast
@pytest.mark.parametrize(
    "length,max_seq_length,eos_token_id,expected_token_ids,expected_token_type_ids",
    [
        (2, 6, -1, [0, 1, -1, -1, -1, -1], [0, -1, 2, 2, 2, 2]),
        (0, 6, -1, [-1, -1, -1, -1, -1, -1], [TokenTypeIds.PADDING] * 6),
    ],
)
def test_pad(
    tokenized_line: TokenizedLine,
    expected_token_ids: List[int],
    expected_token_type_ids: List[int],
):
    """Verify that the tokenized line can be padded with Padding token_ids"""
    assert isinstance(tokenized_line, TokenizedSequence)
    tokenized_line.pad()
    assert tokenized_line.token_ids == expected_token_ids
    assert tokenized_line.token_type_ids == expected_token_type_ids


@pytest.mark.fast
@pytest.mark.parametrize(
    "length,length_2,max_seq_length,max_seq_length_2,eos_token_id,eos_token_id_2",
    [(9, 2, 12, 12, -1, -1), (1, 9, 2, None, -1, None)],
)
def test_pack(
    tokenized_line: TokenizedLine,
    tokenized_line_2: TokenizedLine,
    length: int,
    length_2: int,
    max_seq_length: int,
):
    """Test that one tokenized line can be packed into another tokenized line."""
    assert isinstance(tokenized_line, TokenizedSequence)
    orig_tokenized_line_len = len(tokenized_line)
    new_tokenized_line_len = min(max_seq_length, len(tokenized_line) + len(tokenized_line_2))
    remainder_line = tokenized_line.pack(tokenized_line_2)
    if new_tokenized_line_len == max_seq_length:
        assert tokenized_line.is_packed()
    assert len(tokenized_line) == new_tokenized_line_len
    assert len(remainder_line) + new_tokenized_line_len == orig_tokenized_line_len + len(tokenized_line_2)


def test_get_empty():
    """Test that we can create empty tokenized articles and empty tokenized sequences."""
    empty_article = TokenizedArticle.get_empty()
    empty_sequence = TokenizedSequence.get_empty(4, -1)
    assert empty_article.is_empty()
    assert empty_sequence.is_empty()
