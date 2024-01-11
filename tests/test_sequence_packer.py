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

from typing import List, Tuple

import pytest

from generative_data_prep.processors import Metrics, SequencePacker
from generative_data_prep.tokenized_line import (
    Token,
    TokenizedArticle,
    TokenizedLine,
    TokenizedSequence,
)
from generative_data_prep.utils import (
    OverflowType,
    PackingConfig,
    PackingStyleType,
    TokenTypeIds,
)


@pytest.fixture
def packing_config(packing_style: PackingStyleType, overflow_type: OverflowType):
    """Return a packing config object."""
    return PackingConfig(packing_style, overflow_type)


@pytest.fixture
def sequence_packer(max_seq_length: int, pad_token_id, packing_config: PackingConfig) -> SequencePacker:
    """Create the sequence packer."""
    return SequencePacker(max_seq_length, pad_token_id, packing_config, Metrics())


@pytest.mark.parametrize(
    "length,max_seq_length,eos_token_id,packing_style,overflow_type,length_2,max_seq_length_2,eos_token_id_2,\
        expected_token_ids",
    [
        (3, 5, -1, PackingStyleType.SINGLE, OverflowType.DROP, 6, None, None, []),
        (
            3,
            5,
            -1,
            PackingStyleType.SINGLE,
            OverflowType.TRUNCATE_LEFT,
            6,
            None,
            None,
            [4, 5],
        ),
        (
            3,
            5,
            -1,
            PackingStyleType.SINGLE,
            OverflowType.TRUNCATE_RIGHT,
            6,
            None,
            None,
            [0, 1],
        ),
    ],
)
def test_handle_overflow(
    sequence_packer: SequencePacker,
    tokenized_line: TokenizedLine,
    tokenized_line_2: TokenizedLine,
    expected_token_ids: List[int],
):
    """Test handling overflow function in sequence packer."""
    unfinished_sequence, tokenized_article = tokenized_line, tokenized_line_2
    assert isinstance(unfinished_sequence, TokenizedSequence)
    assert isinstance(tokenized_article, TokenizedArticle)
    tokenized_article = sequence_packer._handle_overflow(tokenized_article, unfinished_sequence)
    assert tokenized_article.dump_token_ids() == expected_token_ids


@pytest.mark.fast
@pytest.mark.parametrize(
    "max_seq_length,eos_token_id,packing_style,overflow_type,article_lengths,expected",
    [
        (
            5,
            -1,
            PackingStyleType.SINGLE,
            OverflowType.DROP,
            [3, 2, 10, 2, 2, 5],
            [(3, 2), (2, 3), (2, 3), (2, 3), (5, 0)],
        ),
        (
            5,
            -1,
            PackingStyleType.SINGLE,
            OverflowType.TRUNCATE_LEFT,
            [3, 2, 10, 2, 2, 5],
            [(3, 2), (2, 3), (5, 0), (2, 3), (2, 3), (5, 0)],
        ),
        (
            5,
            -1,
            PackingStyleType.SINGLE,
            OverflowType.TRUNCATE_RIGHT,
            [3, 2, 10, 2, 2, 5],
            [(3, 2), (2, 3), (5, 0), (2, 3), (2, 3), (5, 0)],
        ),
        (
            5,
            -1,
            PackingStyleType.GREEDY,
            OverflowType.DROP,
            [3, 2, 10, 2, 2, 5],
            [(5, 0), (4, 1), (5, 0)],
        ),
        # all sequences should be dropped test case
        (
            5,
            -1,
            PackingStyleType.GREEDY,
            OverflowType.DROP,
            [6, 6, 6, 6],
            [],
        ),
        (
            5,
            -1,
            PackingStyleType.GREEDY,
            OverflowType.TRUNCATE_LEFT,
            [3, 2, 10, 2, 2, 5],
            [(5, 0), (5, 0), (4, 1), (5, 0)],
        ),
        (
            5,
            -1,
            PackingStyleType.GREEDY,
            OverflowType.TRUNCATE_RIGHT,
            [3, 2, 10, 2, 2, 5],
            [(5, 0), (5, 0), (4, 1), (5, 0)],
        ),
        (
            5,
            -1,
            PackingStyleType.FULL,
            None,
            [3, 2, 10, 2, 2, 5],
            [(5, 0), (5, 0), (5, 0), (5, 0), (4, 1)],
        ),
    ],
)
def test_sequence_packer(
    sequence_packer: SequencePacker,
    max_seq_length: int,
    eos_token_id: int,
    article_lengths: List[int],
    expected: List[Tuple[int, int]],
):
    """Test that the tokenized articles can be packed into fixed length sequences.
    NOTE: ``expected`` should be formatted as a list of tuples, where each tuple is
    (number_of_completion_tokens, number_of_padding_tokens) for a particular sequence.  In addition,
    number_of_completion_tokens + number_of_padding_tokens == max_seq_length must be True.
    """
    err_msg = "Test is invalid, read the note in the docstring"
    assert all(max_seq_length == sum(expected_tuple) for expected_tuple in expected), err_msg

    # create the articles
    articles = []
    for length in article_lengths:
        tokens = []
        for token_id, token_type_id in zip(list(range(length)), [TokenTypeIds.COMPLETION.value] * length):
            tokens.append(Token(token_id, token_type_id))

        articles.append(TokenizedArticle(tokens))
    # pack the sequences
    sequences = sequence_packer(articles)
    assert sequences is not None
    unfinished_sequences = sequence_packer(None)
    sequences += unfinished_sequences
    assert len(sequences) == len(expected)
    assert all(len(sequence) == max_seq_length for sequence in sequences)
    for sequence, (num_compl_tokens, num_pad_tokens) in zip(sequences, expected):
        for i in range(num_compl_tokens):
            assert sequence.tokens[i].token_type_id == TokenTypeIds.COMPLETION.value
        for i in range(num_pad_tokens):
            assert sequence.tokens[i + num_compl_tokens] == Token(eos_token_id, TokenTypeIds.PADDING)
