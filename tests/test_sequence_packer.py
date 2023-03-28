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

import pytest
from typing import List, Tuple

from generative_data_prep.processors import SequencePacker
from generative_data_prep.tokenized_line import TokenizedArticle
from generative_data_prep.utils import PackingStyleType, TokenTypeIds


@pytest.fixture
def sequence_packer(max_seq_length: int, eos_token_id: int, packing_style: PackingStyleType) -> SequencePacker:
    """Create the sequence packer."""
    return SequencePacker(max_seq_length, eos_token_id, packing_style)


@pytest.mark.parametrize('max_seq_length,eos_token_id,packing_style,article_lengths,expected', [
    (5, -1, PackingStyleType.SINGLE_DROP_OVERFLOW, [3, 2, 10, 2, 2, 5], [(3, 2), (2, 3), (2, 3), (2, 3), (5, 0)]),
    (5, -1, PackingStyleType.SINGLE_TRUNCATE_OVERFLOW, [3, 2, 10, 2, 2, 5], [(3, 2), (2, 3), (5, 0), (2, 3), (2, 3),
                                                                             (5, 0)]),
    (5, -1, PackingStyleType.GREEDY, [3, 2, 10, 2, 2, 5], [(5, 0), (4, 1), (5, 0)]),
    (5, -1, PackingStyleType.FULL, [3, 2, 10, 2, 2, 5], [(5, 0), (5, 0), (5, 0), (5, 0), (4, 1)]),
])
def test_sequence_packer(sequence_packer: SequencePacker, max_seq_length: int, eos_token_id: int,
                         article_lengths: List[int], expected: List[Tuple[int, int]]):
    """Test that the tokenized articles can be packed into fixed length sequences.

    NOTE: ``expected`` should be formatted as a list of tuples, where each tuple is
    (number_of_completion_tokens, number_of_padding_tokens) for a particular sequence.  In addition,
    number_of_completion_tokens + number_of_padding_tokens == max_seq_length must be True.
    """
    err_msg = 'Test is invalid, read the note in the docstring'
    assert all(max_seq_length == sum(expected_tuple) for expected_tuple in expected), err_msg

    # create the articles
    articles = []
    for length in article_lengths:
        tokens = list(range(length))
        token_type_ids = [TokenTypeIds.COMPLETION.value] * length
        articles.append(TokenizedArticle(tokens, token_type_ids))

    # pack the sequences
    sequences = sequence_packer(articles)
    assert sequences is not None
    unfinished_sequences = sequence_packer(None)
    sequences += unfinished_sequences

    assert len(sequences) == len(expected)
    assert all(len(sequence) == max_seq_length for sequence in sequences)
    for sequence, (num_compl_tokens, num_pad_tokens) in zip(sequences, expected):
        for i in range(num_compl_tokens):
            assert sequence.token_type_ids[i] == TokenTypeIds.COMPLETION.value
        for i in range(num_pad_tokens):
            assert sequence[i + num_compl_tokens] == (eos_token_id, TokenTypeIds.PADDING)
