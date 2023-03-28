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

import os
import pytest
from transformers import GPT2Tokenizer
from typing import List, Optional, Union

from generative_data_prep.processors import ArticleTokenizer
from generative_data_prep.tokenized_line import TokenizedSequence
from generative_data_prep.utils import BoundaryType, TokenTypeIds

TOKENIZER = GPT2Tokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gpt2_tokenizer'))
MAX_SEQ_LEN = 6
EOS_TOKEN_ID = 0
SEP = TokenTypeIds.SEP
PROMPT = TokenTypeIds.PROMPT
COMP = TokenTypeIds.COMPLETION
PAD = TokenTypeIds.PADDING


def mock_tokenize(input_text: str) -> List[int]:
    FAKE_TOKENIZER = {"hi": 1, "bye": 2, "test": 3}
    mock_token_ids = []
    for word in input_text.split(' '):
        if word:
            mock_token_ids.append(FAKE_TOKENIZER[word])

    return mock_token_ids


@pytest.fixture
def article_tokenizer(monkeypatch, packing_boundary: BoundaryType) -> ArticleTokenizer:
    """Creates a tokenized line."""
    monkeypatch.setattr(TOKENIZER, "encode", mock_tokenize)
    article_tokenizer = ArticleTokenizer(TOKENIZER, MAX_SEQ_LEN, packing_boundary=packing_boundary)
    monkeypatch.setattr(article_tokenizer, "eos_token_id", EOS_TOKEN_ID)
    monkeypatch.setattr(article_tokenizer.packer, "eos_token_id", EOS_TOKEN_ID)
    return article_tokenizer


@pytest.mark.parametrize(
    'packing_boundary,prompt,completion,gold_token_ids,gold_ttids',
    [
        (BoundaryType.JSONL, "", "", [], []),  # empty prompt and completion
        (BoundaryType.JSONL, "", "bye", [2, 0], [COMP, COMP]),  # empty prompt
        (BoundaryType.JSONL, "hi", "", [1, 0], [PROMPT, COMP]),  # empty completion
        (BoundaryType.JSONL, "hi", "bye", [1, 2, 0], [PROMPT, COMP, COMP]),  # full prompt and completion
        (BoundaryType.JSONL, "hi test", "bye test", [1, 3, 2, 3, 0], [PROMPT, PROMPT, COMP, COMP, COMP])
    ])
def test_tokenize(article_tokenizer: ArticleTokenizer, prompt: str, completion: str, gold_token_ids: List[int],
                  gold_ttids: List[int]):
    token_ids, ttids = article_tokenizer.tokenize(completion, prompt)

    assert token_ids == gold_token_ids
    assert ttids == gold_ttids


@pytest.mark.parametrize(
    'packing_boundary,prompt,completion,gold_prompt,gold_completion',
    [
        (BoundaryType.JSONL, "hi", "bye", "hi", " bye"),  # no spaces
        (BoundaryType.JSONL, " hi", "bye", " hi", " bye"),  # space in front of prompt
        (BoundaryType.JSONL, "hi ", "bye", "hi", " bye"),  # space after prompt
        (BoundaryType.JSONL, "hi", " bye", "hi", " bye"),  # space before completion
        (BoundaryType.JSONL, "hi", "bye ", "hi", " bye "),  # space after completion
        (BoundaryType.JSONL, "hi ", " bye", "hi", " bye"),  # space after prompt before completion
        (BoundaryType.JSONL, " h i ", " b y e ", " h i", " b y e ")  # spaces in other places
    ])
def test_add_space_separator(article_tokenizer: ArticleTokenizer, prompt: str, completion: str, gold_prompt: List[int],
                             gold_completion: List[int]):
    completion_test, prompt_test = article_tokenizer._add_space_separator(completion, prompt)

    assert completion_test == gold_completion
    assert prompt_test == gold_prompt


@pytest.mark.parametrize(
    'packing_boundary,jsonl,gold_token_ids,gold_ttids',
    [
        (BoundaryType.JSONL, {
            "prompt": "",
            "completion": ""
        }, [], []),  # single jsonl empty
        (BoundaryType.JSONL, {
            "prompt": "hi",
            "completion": ""
        }, [[1, 0]], [[PROMPT, SEP]]),
        (BoundaryType.JSONL, {
            "prompt": "",
            "completion": "bye"
        }, [[2, 0]], [[COMP, SEP]]),
        (BoundaryType.JSONL, {
            "prompt": "hi test",
            "completion": "bye test"
        }, [[1, 3, 2, 3, 0]], [[PROMPT, PROMPT, COMP, COMP, SEP]]),  # single jsonl prompt and completion
        (BoundaryType.JSONL, [{
            "prompt": "hi",
            "completion": "bye"
        }], [[1, 2, 0]], [[PROMPT, COMP, SEP]]),  # list length 1
        (BoundaryType.JSONL, [{
            "prompt": "hi",
            "completion": "bye"
        }, {
            "prompt": "test",
            "completion": "test"
        }], [[1, 2, 0, 3, 3, 0]], [[PROMPT, COMP, COMP, PROMPT, COMP, SEP]]),  # list length 2
        (BoundaryType.PROMPT_COMPLETION_PAIR, {
            "prompt": "",
            "completion": ""
        }, [], []),
        (BoundaryType.PROMPT_COMPLETION_PAIR, {
            "prompt": "hi",
            "completion": ""
        }, [[1, 0]], [[PROMPT, SEP]]),
        (BoundaryType.PROMPT_COMPLETION_PAIR, {
            "prompt": "",
            "completion": "bye"
        }, [[2, 0]], [[COMP, SEP]]),
        (BoundaryType.PROMPT_COMPLETION_PAIR, {
            "prompt": "hi test",
            "completion": "bye test"
        }, [[1, 3, 2, 3, 0]], [[PROMPT, PROMPT, COMP, COMP, SEP]]),
        (BoundaryType.PROMPT_COMPLETION_PAIR, [{
            "prompt": "hi",
            "completion": "bye"
        }], [[1, 2, 0]], [[PROMPT, COMP, SEP]]),
        (BoundaryType.PROMPT_COMPLETION_PAIR, [{
            "prompt": "hi",
            "completion": "bye"
        }, {
            "prompt": "test",
            "completion": "test"
        }], [[1, 2, 0], [3, 3, 0]], [[PROMPT, COMP, COMP], [PROMPT, COMP, SEP]])
    ])
def test_process_jsonl(article_tokenizer: ArticleTokenizer, jsonl: Union[dict, List], gold_token_ids: List[List[int]],
                       gold_ttids: List[List[int]]):
    tokenized_articles = article_tokenizer.process_jsonl(jsonl)
    for tokenized_article, gold_token, gold_ttid in zip(tokenized_articles, gold_token_ids, gold_ttids):
        assert tokenized_article.token_ids == gold_token
        assert tokenized_article.token_type_ids == gold_ttid


@pytest.mark.parametrize(
    'packing_boundary,text,gold_token_ids,gold_ttids',
    [
        (BoundaryType.JSONL, "", [], []),  # empty text
        (BoundaryType.JSONL, "hi", [1, 0], [COMP, SEP]),  # one word
        (BoundaryType.JSONL, "hi bye test", [1, 2, 3, 0], [COMP, COMP, COMP, SEP]),  # multiple word
    ])
def test_process_text(article_tokenizer: ArticleTokenizer, text: str, gold_token_ids: List[int], gold_ttids: List[int]):
    tokenized_articles = article_tokenizer.process_text(text)
    assert tokenized_articles[0].token_ids == gold_token_ids
    assert tokenized_articles[0].token_type_ids == gold_ttids


def get_tokenized_seq(token_ids: List[int], token_type_ids: List[int]) -> TokenizedSequence:
    return TokenizedSequence(token_ids, token_type_ids, MAX_SEQ_LEN, EOS_TOKEN_ID)


@pytest.mark.parametrize(
    'packing_boundary,article,gold_tokenized_sequence,gold_unfinished_sequence',
    [
        (BoundaryType.JSONL, None, [], get_tokenized_seq([], [])),  # test None passed in
        (BoundaryType.JSONL, "hi bye test", [], get_tokenized_seq([1, 2, 3, 0], [COMP, COMP, COMP, SEP])),
        (BoundaryType.JSONL, '{"prompt": "hi", "completion": "bye"}', [],
         get_tokenized_seq([1, 2, 0], [PROMPT, COMP, SEP])),
        (BoundaryType.JSONL, '[{"prompt": "hi", "completion": "bye"}, {"prompt": "test", "completion": "test"}, \
        {"prompt": "test", "completion": "test"}]', [
            get_tokenized_seq([1, 2, 0, 3, 3, 0], [PROMPT, COMP, COMP, PROMPT, COMP, COMP])
        ], get_tokenized_seq([3, 3, 0], [PROMPT, COMP, SEP]))
    ])
def test__call__(article_tokenizer: ArticleTokenizer, article: Optional[str],
                 gold_tokenized_sequence: List[TokenizedSequence], gold_unfinished_sequence: List[TokenizedSequence]):
    tokenized_sequence = article_tokenizer(article)
    assert tokenized_sequence == gold_tokenized_sequence
    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence


@pytest.mark.parametrize('packing_boundary,articles,gold_tokenized_sequences,gold_unfinished_sequence',
                         [(BoundaryType.JSONL, ["hi bye", "hi bye", "hi hi"], [
                             [], [], [get_tokenized_seq([1, 2, 0, 1, 2, 0], [COMP, COMP, SEP, COMP, COMP, SEP])]
                         ], get_tokenized_seq([1, 1, 0], [COMP, COMP, SEP])),
                          (BoundaryType.JSONL, [
                              '{"prompt": "hi", "completion": "bye"}', '{"prompt": "hi", "completion": "bye"}', '{ \
        "prompt": "hi", "completion": "hi"}'
                          ], [[], [], [get_tokenized_seq([1, 2, 0, 1, 2, 0], [PROMPT, COMP, SEP, PROMPT, COMP, SEP])]
                              ], get_tokenized_seq([1, 1, 0], [PROMPT, COMP, SEP])),
                          (BoundaryType.JSONL, ["hi bye", '{"prompt": "hi test", "completion": "bye test"}', None], [
                              [], [get_tokenized_seq([1, 2, 0, 1, 3, 2], [COMP, COMP, SEP, PROMPT, PROMPT, COMP])],
                              [get_tokenized_seq([3, 0, 0, 0, 0, 0], [COMP, SEP, PAD, PAD, PAD, PAD])]
                          ], get_tokenized_seq([], []))])
def test_multiple__call__(article_tokenizer: ArticleTokenizer, articles: List[Optional[str]],
                          gold_tokenized_sequences: List[List[TokenizedSequence]],
                          gold_unfinished_sequence: List[TokenizedSequence]):
    for article, gold_tokenized_sequence in zip(articles, gold_tokenized_sequences):
        tokenized_sequence = article_tokenizer(article)
        assert tokenized_sequence == gold_tokenized_sequence

    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence
