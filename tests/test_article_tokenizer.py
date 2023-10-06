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

from typing import List, Optional, Union

import pytest
from transformers import GPT2Tokenizer

from generative_data_prep.processors import ArticleTokenizer
from generative_data_prep.tokenized_line import Token, TokenizedSequence
from generative_data_prep.utils import BoundaryType, FileExtension, TokenTypeIds

TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
MAX_SEQ_LEN = 6
EOS_TOKEN_ID = 0
SEP = TokenTypeIds.SEP
PROMPT = TokenTypeIds.PROMPT
COMP = TokenTypeIds.COMPLETION
PAD = TokenTypeIds.PADDING


def mock_tokenize(input_text: str) -> List[int]:
    """Fake tokenizer that takes in fake text and assigns fake token ids."""
    FAKE_TOKENIZER = {"hi": 1, "bye": 2, "test": 3, "<human>": 4, "<bot>": 5}
    mock_token_ids = []
    for word in input_text.split(" "):
        if word:
            mock_token_ids.append(FAKE_TOKENIZER[word])

    return mock_token_ids


@pytest.fixture
def article_tokenizer(
    monkeypatch,
    packing_boundary: BoundaryType,
    ext_type: FileExtension,
    prompt_prefix: str,
    prompt_postfix: str,
) -> ArticleTokenizer:
    """Creates a tokenized line."""
    monkeypatch.setattr(TOKENIZER, "encode", mock_tokenize)
    article_tokenizer = ArticleTokenizer(
        TOKENIZER,
        MAX_SEQ_LEN,
        ext_type,
        packing_boundary=packing_boundary,
        prompt_prefix=prompt_prefix,
        prompt_postfix=prompt_postfix,
    )
    monkeypatch.setattr(article_tokenizer, "eos_token_id", EOS_TOKEN_ID)
    monkeypatch.setattr(article_tokenizer.packer, "eos_token_id", EOS_TOKEN_ID)
    return article_tokenizer


@pytest.fixture
def article_tokenizer_for_prompt_sequences(
    monkeypatch,
    packing_boundary: BoundaryType,
    ext_type: FileExtension,
    keep_prompt_only_sequences: bool,
) -> ArticleTokenizer:
    """Creates a tokenized line."""
    monkeypatch.setattr(TOKENIZER, "encode", mock_tokenize)
    article_tokenizer = ArticleTokenizer(
        TOKENIZER,
        MAX_SEQ_LEN,
        ext_type,
        packing_boundary=packing_boundary,
        keep_prompt_only_sequences=keep_prompt_only_sequences,
    )
    monkeypatch.setattr(article_tokenizer, "eos_token_id", EOS_TOKEN_ID)
    monkeypatch.setattr(article_tokenizer.packer, "eos_token_id", EOS_TOKEN_ID)
    return article_tokenizer


@pytest.mark.fast
@pytest.mark.parametrize(
    "packing_boundary,ext_type,prompt_prefix,prompt_postfix,prompt,completion,gold_token_ids,gold_ttids",
    [
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "",
            "",
            [],
            [],
        ),  # empty prompt and completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "",
            "bye",
            [2, 0],
            [COMP, COMP],
        ),  # empty prompt
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi",
            "",
            [1, 0],
            [PROMPT, COMP],
        ),  # empty completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi",
            "bye",
            [1, 2, 0],
            [PROMPT, COMP, COMP],
        ),  # full prompt and completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi test",
            "bye test",
            [1, 3, 2, 3, 0],
            [PROMPT, PROMPT, COMP, COMP, COMP],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            "<human> ",
            " <bot>",
            "hi test",
            "bye test",
            [4, 1, 3, 5, 2, 3, 0],
            [PROMPT, PROMPT, PROMPT, PROMPT, COMP, COMP, COMP],
        ),  # full prompt and completion with chat-ml tag
    ],
)
def test_tokenize(
    article_tokenizer: ArticleTokenizer,
    prompt: str,
    completion: str,
    gold_token_ids: List[int],
    gold_ttids: List[int],
):
    """Test the tokenize function."""
    tokens = article_tokenizer.tokenize(completion, prompt)

    assert list(map(lambda x: x.token_id, tokens)) == gold_token_ids
    assert list(map(lambda x: x.token_type_id, tokens)) == gold_ttids


@pytest.mark.parametrize(
    "packing_boundary,ext_type,prompt_prefix,prompt_postfix,prompt,completion,gold_prompt,gold_completion",
    [
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi",
            "bye",
            "hi",
            " bye",
        ),  # no spaces
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            " hi",
            "bye",
            " hi",
            " bye",
        ),  # space in front of prompt
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi ",
            "bye",
            "hi",
            " bye",
        ),  # space after prompt
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi",
            " bye",
            "hi",
            " bye",
        ),  # space before completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi",
            "bye ",
            "hi",
            " bye ",
        ),  # space after completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            "hi ",
            " bye",
            "hi",
            " bye",
        ),  # space after prompt before completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            " h i ",
            " b y e ",
            " h i",
            " b y e ",
        ),  # spaces in other places
    ],
)
def test_add_space_separator(
    article_tokenizer: ArticleTokenizer,
    prompt: str,
    completion: str,
    gold_prompt: List[int],
    gold_completion: List[int],
):
    """Test adding the add_space_separator function."""
    completion_test, prompt_test = article_tokenizer._add_space_separator(completion, prompt)

    assert completion_test == gold_completion
    assert prompt_test == gold_prompt


@pytest.mark.parametrize(
    "packing_boundary,ext_type,keep_prompt_only_sequences,jsonl,gold_token_ids,gold_ttids",
    [
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            {"prompt": "", "completion": ""},
            [],
            [],
        ),  # single jsonl empty
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            {"prompt": "hi", "completion": ""},
            [[1, 0]],
            [[PROMPT, SEP]],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            {"prompt": "", "completion": "bye"},
            [[2, 0]],
            [[COMP, SEP]],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            {"prompt": "hi test", "completion": "bye test"},
            [[1, 3, 2, 3, 0]],
            [[PROMPT, PROMPT, COMP, COMP, SEP]],
        ),  # single jsonl prompt and completion
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            [{"prompt": "hi", "completion": "bye"}],
            [[1, 2, 0]],
            [[PROMPT, COMP, SEP]],
        ),  # list length 1
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            [
                {"prompt": "hi", "completion": "bye"},
                {"prompt": "test", "completion": "test"},
            ],
            [[1, 2, 0, 3, 3, 0]],
            [[PROMPT, COMP, COMP, PROMPT, COMP, SEP]],
        ),  # list length 2
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            {"prompt": "", "completion": ""},
            [],
            [],
        ),
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            {"prompt": "hi", "completion": ""},
            [[1, 0]],
            [[PROMPT, SEP]],
        ),
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            {"prompt": "", "completion": "bye"},
            [[2, 0]],
            [[COMP, SEP]],
        ),
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            {"prompt": "hi test", "completion": "bye test"},
            [[1, 3, 2, 3, 0]],
            [[PROMPT, PROMPT, COMP, COMP, SEP]],
        ),
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            [{"prompt": "hi", "completion": "bye"}],
            [[1, 2, 0]],
            [[PROMPT, COMP, SEP]],
        ),
        (
            BoundaryType.PROMPT_COMPLETION_PAIR,
            FileExtension.JSONL,
            True,
            [
                {"prompt": "hi", "completion": "bye"},
                {"prompt": "test", "completion": "test"},
            ],
            [[1, 2, 0], [3, 3, 0]],
            [[PROMPT, COMP, COMP], [PROMPT, COMP, SEP]],
        ),
    ],
)
def test_process_jsonl(
    article_tokenizer_for_prompt_sequences: ArticleTokenizer,
    jsonl: Union[dict, List],
    gold_token_ids: List[List[int]],
    gold_ttids: List[List[int]],
):
    """Test process_jsonl function to make sure it correctly process jsonl."""
    tokenized_articles = article_tokenizer_for_prompt_sequences.process_jsonl(jsonl)
    for tokenized_article, gold_token, gold_ttid in zip(tokenized_articles, gold_token_ids, gold_ttids):
        assert tokenized_article.dump_token_ids() == gold_token
        assert tokenized_article.dump_token_type_ids() == gold_ttid


@pytest.mark.parametrize(
    "packing_boundary,ext_type,prompt_prefix,prompt_postfix,text,gold_token_ids,gold_ttids",
    [
        (BoundaryType.JSONL, FileExtension.JSONL, None, None, "", [], []),  # empty text
        (
            BoundaryType.JSONL,
            FileExtension.TXT,
            None,
            None,
            "hi",
            [1, 0],
            [COMP, SEP],
        ),  # one word
        (
            BoundaryType.JSONL,
            FileExtension.TXT,
            None,
            None,
            "hi bye test",
            [1, 2, 3, 0],
            [COMP, COMP, COMP, SEP],
        ),  # multiple word
    ],
)
def test_process_text(
    article_tokenizer: ArticleTokenizer,
    text: str,
    gold_token_ids: List[int],
    gold_ttids: List[int],
):
    """Test process text to make sure it returns the correct tokenized articles."""
    tokenized_articles = article_tokenizer.process_text(text)
    assert tokenized_articles[0].dump_token_ids() == gold_token_ids
    assert tokenized_articles[0].dump_token_type_ids() == gold_ttids


def get_tokenized_seq(tokens: List[Token]) -> TokenizedSequence:
    """Return a toeknized sequence object version of given token ids and ttids."""
    return TokenizedSequence(tokens, MAX_SEQ_LEN, EOS_TOKEN_ID)


@pytest.mark.parametrize(
    "packing_boundary,ext_type,prompt_prefix,prompt_postfix,article,gold_tokenized_sequence,gold_unfinished_sequence",
    [
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            None,
            [],
            get_tokenized_seq([]),
        ),  # test None passed in
        (
            BoundaryType.JSONL,
            FileExtension.TXT,
            None,
            None,
            "hi bye test",
            [],
            get_tokenized_seq([Token(1, COMP), Token(2, COMP), Token(3, COMP), Token(0, SEP)]),
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            '{"prompt": "hi", "completion": "bye"}',
            [],
            get_tokenized_seq([Token(1, PROMPT), Token(2, COMP), Token(0, SEP)]),
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            '[{"prompt": "hi", "completion": "bye"}, {"prompt": "test", "completion": "test"}, \
        {"prompt": "test", "completion": "test"}]',
            [
                get_tokenized_seq(
                    [Token(1, PROMPT), Token(2, COMP), Token(0, COMP), Token(3, PROMPT), Token(3, COMP), Token(0, COMP)]
                )
            ],
            get_tokenized_seq([Token(3, PROMPT), Token(3, COMP), Token(0, SEP)]),
        ),
    ],
)
def test__call__(
    article_tokenizer: ArticleTokenizer,
    article: Optional[str],
    gold_tokenized_sequence: List[TokenizedSequence],
    gold_unfinished_sequence: List[TokenizedSequence],
):
    """Test calling article tokenizer."""
    tokenized_sequence = article_tokenizer(article)
    assert tokenized_sequence == gold_tokenized_sequence
    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence


@pytest.mark.parametrize(
    "packing_boundary,ext_type,prompt_prefix,prompt_postfix,articles,gold_tokenized_sequences,gold_unfinished_sequence",
    [
        (
            BoundaryType.JSONL,
            FileExtension.TXT,
            None,
            None,
            ["hi bye", "hi bye", "hi hi"],
            [
                [],
                [],
                [
                    get_tokenized_seq(
                        [Token(1, COMP), Token(2, COMP), Token(0, SEP), Token(1, COMP), Token(2, COMP), Token(0, SEP)]
                    )
                ],
            ],
            get_tokenized_seq([Token(1, COMP), Token(1, COMP), Token(0, SEP)]),
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            None,
            None,
            [
                '{"prompt": "hi", "completion": "bye"}',
                '{"prompt": "hi", "completion": "bye"}',
                '{ \
        "prompt": "hi", "completion": "hi"}',
            ],
            [
                [],
                [],
                [
                    get_tokenized_seq(
                        [
                            Token(1, PROMPT),
                            Token(2, COMP),
                            Token(0, SEP),
                            Token(1, PROMPT),
                            Token(2, COMP),
                            Token(0, SEP),
                        ]
                    )
                ],
            ],
            get_tokenized_seq([Token(1, PROMPT), Token(1, COMP), Token(0, SEP)]),
        ),
    ],
)
def test_multiple__call__(
    article_tokenizer: ArticleTokenizer,
    articles: List[Optional[str]],
    gold_tokenized_sequences: List[List[TokenizedSequence]],
    gold_unfinished_sequence: List[TokenizedSequence],
):
    """Test calling article tokenizer multiple times."""
    for article, gold_tokenized_sequence in zip(articles, gold_tokenized_sequences):
        tokenized_sequence = article_tokenizer(article)
        assert tokenized_sequence == gold_tokenized_sequence

    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence


@pytest.mark.parametrize(
    "packing_boundary,ext_type,keep_prompt_only_sequences,articles,gold_token_ids,gold_token_type_ids",
    [
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            False,
            '[{"prompt": "hi bye test hi bye", "completion": ""}]',
            [],
            [],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            '[{"prompt": "hi bye test hi bye", "completion": ""}]',
            [[1, 2, 3, 1, 2, 0]],
            [[PROMPT, PROMPT, PROMPT, PROMPT, PROMPT, SEP]],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            False,
            '[{"prompt": "hi", "completion": ""}, \
      {"prompt": "hi bye", "completion": ""}, \
      {"prompt": "hi bye test", "completion": ""}]',
            [],
            [],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            False,
            '[{"prompt": "hi hi hi hi hi hi hi hi", "completion": "bye"}]',
            [[1, 1, 2, 0, 0, 0]],
            [[PROMPT, PROMPT, COMP, SEP, PAD, PAD]],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            True,
            '[{"prompt": "hi hi hi hi hi hi hi hi", "completion": "bye"}]',
            [[1, 1, 1, 1, 1, 1], [1, 1, 2, 0, 0, 0]],
            [
                [PROMPT, PROMPT, PROMPT, PROMPT, PROMPT, PROMPT],
                [PROMPT, PROMPT, COMP, SEP, PAD, PAD],
            ],
        ),
        (
            BoundaryType.JSONL,
            FileExtension.JSONL,
            False,
            '[{"prompt": "hi bye test hi bye test", "completion": ""}, \
            {"prompt": "hi bye test hi", "completion": "test hi"}]',
            [[1, 2, 3, 1, 3, 1]],
            [[PROMPT, PROMPT, PROMPT, PROMPT, COMP, COMP]],
        ),
    ],
)
def test_prompt_only_sequences(
    article_tokenizer_for_prompt_sequences: ArticleTokenizer,
    articles: Optional[str],
    gold_token_ids: List[int],
    gold_token_type_ids: List[int],
):
    """Test using only prompt sequences to make sure they are dropped."""
    sequences = []
    sequences += article_tokenizer_for_prompt_sequences(articles)
    sequences += article_tokenizer_for_prompt_sequences(None)
    token_ids = [seq.dump_token_ids() for seq in sequences]
    token_tids = [seq.dump_token_type_ids() for seq in sequences]
    assert token_ids == gold_token_ids
    assert token_tids == gold_token_type_ids
