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


This module implements a SequencePacker.

The Sequence Packer takes tokenized articles and 'packs' or compresses them into fixed length tokenized sequences.
The reason for this is because our models need to train on fixed length inputs, but because text can be arbitrary
long, we need to sometimes split up long blocks of text into multiple sequences.
"""

from typing import List, Optional, Tuple

from generative_data_prep.tokenized_line import TokenizedArticle, TokenizedSequence
from generative_data_prep.utils import OverflowType, PackingConfig, PackingStyleType


class SequencePacker:
    """Takes articles and packs them into fixed length sequences."""

    def __init__(
        self, max_seq_length: int, eos_token_id: int, packing_config: PackingConfig
    ):
        """Create the SequencePacker.

        Args:
            max_seq_length:  The maximum length of the tokenized sequences.
            eos_token_id:   The end of text token.  This is used by the tokenized sequences for padding.
            packing_style:  The method we use to 'pack' the tokenized articles into tokenized sequences.  We currently
                support 4 different packing styles:
                    FULL:  Completely fill sequences with tokens, as soon as sequences is full start packing into
                        new sequence. Ignore article boundaries, they may be split across multiple sequences
                    GREEDY:  Fit as many articles as possible into a sequence, make sure no article is split
                        across multiple sequences. Fill the left over space in each sequence with padding.
                        If an article is longer than the maximum sequence length, throw it out.
                    SINGLE_DROP_OVERFLOW:  Each sequence contains only 1 article.  Fill the rest of the sequence with
                        padding.  If an article is longer than the maximum sequence length, throw out the article.
                    SINGLE_TRUNCATE_OVERFLOW:  Each sequence contains only 1 article.  Fill the rest of the sequence
                        with padding.  If an article is longer than the maximum sequence length truncate it, and throw
                        the remaining portion of the article out.

        Example:
            >>> from generative_data_prep.utils import TokenTypeIds
            >>>
            >>> articles = [
            ...     TokenizedArticle([0, 1], [TokenTypeIds.PROMPT] * 2),
            ...     TokenizedArticle([2, 3], [TokenTypeIds.PROMPT] * 2),
            ...     TokenizedArticle([4, 5, 6, 7], [TokenTypeIds.PROMPT] * 4)
            ... ]
            >>> for packing_config in PackingConfig.get_choices():
            ...     sequence_packer = SequencePacker(3, -1, packing_config)
            ...     sequences = sequence_packer(articles)
            ...     sequences += sequence_packer(None)
            ...     sequences_str = '\\n'.join(map(str, sequences))
            ...     print(f'Packing Style {packing_config}\\n{sequences_str}')
            ...
            Packing Style greedy::drop
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            Packing Style greedy::truncate_left
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(5, <TokenTypeIds.PROMPT: 0>) (6, <TokenTypeIds.PROMPT: 0>) (7, <TokenTypeIds.PROMPT: 0>)]
            Packing Style greedy::truncate_right
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(4, <TokenTypeIds.PROMPT: 0>) (5, <TokenTypeIds.PROMPT: 0>) (6, <TokenTypeIds.PROMPT: 0>)]
            Packing Style full
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (2, <TokenTypeIds.PROMPT: 0>)]
            [(3, <TokenTypeIds.PROMPT: 0>) (4, <TokenTypeIds.PROMPT: 0>) (5, <TokenTypeIds.PROMPT: 0>)]
            [(6, <TokenTypeIds.PROMPT: 0>) (7, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            Packing Style single::drop
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            Packing Style single::truncate_left
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(5, <TokenTypeIds.PROMPT: 0>) (6, <TokenTypeIds.PROMPT: 0>) (7, <TokenTypeIds.PROMPT: 0>)]
            Packing Style single::truncate_right
            [(0, <TokenTypeIds.PROMPT: 0>) (1, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(2, <TokenTypeIds.PROMPT: 0>) (3, <TokenTypeIds.PROMPT: 0>) (-1, <TokenTypeIds.PADDING: 2>)]
            [(4, <TokenTypeIds.PROMPT: 0>) (5, <TokenTypeIds.PROMPT: 0>) (6, <TokenTypeIds.PROMPT: 0>)]
        """
        self.eos_token_id = eos_token_id
        self.max_seq_length = max_seq_length
        self.packing_config = packing_config

        self.unfinished_sequence = TokenizedSequence.get_empty(
            self.max_seq_length, self.eos_token_id
        )

    def __call__(
        self, tokenized_articles: Optional[List[TokenizedArticle]]
    ) -> List[TokenizedSequence]:
        """Call the SequencePacker

        Args:
            tokenized_articles:  The tokenized articles that will be packed.

        Returns:
            The resulting tokenized sequences.
        """
        unfinished_sequence = self.unfinished_sequence

        # If no more tokenized articles, return the last unfinished sequence
        if tokenized_articles is None:
            if not self.unfinished_sequence.is_empty():
                unfinished_sequence.pad()
                self.unfinished_sequence = TokenizedSequence.get_empty(
                    self.max_seq_length, self.eos_token_id
                )
                return [unfinished_sequence]
            return []

        # Convert the tokenized article into packed sequences
        packed_sequences = []
        for tokenized_article in tokenized_articles:
            newly_packed_sequences, unfinished_sequence = self._get_packed_sequences(
                tokenized_article, unfinished_sequence
            )
            packed_sequences += newly_packed_sequences

        # Store any unfinished sequences to prepend to the next tokenized article which will be passed in when
        # this SequencePacker is called again.
        self.unfinished_sequence = unfinished_sequence
        return packed_sequences

    def _handle_overflow(
        self,
        tokenized_article: TokenizedArticle,
        unfinished_sequence: TokenizedSequence,
    ):
        num_overflow_tokens = len(tokenized_article) - unfinished_sequence.free_tokens
        if num_overflow_tokens <= 0:
            return tokenized_article

        if self.packing_config.overflow_type == OverflowType.DROP:
            return tokenized_article.get_empty()
        elif self.packing_config.overflow_type == OverflowType.TRUNCATE_LEFT:
            return tokenized_article[num_overflow_tokens:]
        elif self.packing_config.overflow_type == OverflowType.TRUNCATE_RIGHT:
            return tokenized_article[:-num_overflow_tokens]
        else:
            raise ValueError(
                f"Invalid Overflow Type {self.packing_config.overflow_type}"
            )

    def _get_packed_sequences(
        self,
        tokenized_article: TokenizedArticle,
        unfinished_sequence: TokenizedSequence,
    ) -> Tuple[List[TokenizedSequence], TokenizedSequence]:
        """Packs the tokenized article into sequences.
        Args:
            tokenized_article:   The tokenized article to be packed.
            unfinished_sequence: A sequence that is not at its maximum length just yet.  Depending on the packing
                style we might try and fit the tokenized article into this sequence.
        Returns:
            A tuple with the packed sequences and any leftover tokens in an unfinished token sequence
        """
        newly_packed_sequences = []

        if self.packing_config.packing_style == PackingStyleType.SINGLE:
            # Put each article in its own sequence
            assert unfinished_sequence.is_empty()
            tokenized_article = self._handle_overflow(
                tokenized_article, unfinished_sequence
            )
            unfinished_sequence += tokenized_article
            if not unfinished_sequence.is_empty():
                unfinished_sequence.pad()
                newly_packed_sequences.append(unfinished_sequence)
                unfinished_sequence = TokenizedSequence.get_empty(
                    self.max_seq_length, self.eos_token_id
                )

        elif self.packing_config.packing_style == PackingStyleType.GREEDY:
            # if it fits in the unfinished sequence, then add it
            if len(unfinished_sequence) + len(tokenized_article) <= self.max_seq_length:
                unfinished_sequence += tokenized_article
            else:
                # complete the previous sequence
                unfinished_sequence.pad()
                newly_packed_sequences.append(unfinished_sequence)
                # try and fit the tokenized article in the next sequence
                unfinished_sequence = TokenizedSequence.get_empty(
                    self.max_seq_length, self.eos_token_id
                )
                tokenized_article = self._handle_overflow(
                    tokenized_article, unfinished_sequence
                )
                unfinished_sequence += tokenized_article

        elif self.packing_config.packing_style == PackingStyleType.FULL:
            # Stuff the article into as many sequences as is required
            remainder_article = unfinished_sequence.pack(tokenized_article)
            while not remainder_article.is_empty():
                newly_packed_sequences.append(unfinished_sequence)
                unfinished_sequence = TokenizedSequence.get_empty(
                    self.max_seq_length, self.eos_token_id
                )
                remainder_article = unfinished_sequence.pack(remainder_article)

        else:
            raise ValueError(
                f"Invalid packing style {self.packing_config.packing_style}"
            )

        return newly_packed_sequences, unfinished_sequence
