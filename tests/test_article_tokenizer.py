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

from typing import List, Optional, Union

import pytest
from transformers import GPT2Tokenizer

from generative_data_prep.processors import ArticleTokenizer
from generative_data_prep.tokenized_line import TokenizedSequence
from generative_data_prep.utils import (BoundaryType, FileExtension,
                                        TokenTypeIds)

TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
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
def article_tokenizer(monkeypatch, packing_boundary: BoundaryType,
                      ext_type: FileExtension) -> ArticleTokenizer:
    """Creates a tokenized line."""
    monkeypatch.setattr(TOKENIZER, "encode", mock_tokenize)
    article_tokenizer = ArticleTokenizer(TOKENIZER,
                                         MAX_SEQ_LEN,
                                         ext_type,
                                         packing_boundary=packing_boundary)
    monkeypatch.setattr(article_tokenizer, "eos_token_id", EOS_TOKEN_ID)
    monkeypatch.setattr(article_tokenizer.packer, "eos_token_id", EOS_TOKEN_ID)
    return article_tokenizer


@pytest.fixture
def article_tokenizer_for_prompt_sequences(
        monkeypatch, packing_boundary: BoundaryType, ext_type: FileExtension,
        keep_prompt_only_sequences: bool) -> ArticleTokenizer:
    """Creates a tokenized line."""
    monkeypatch.setattr(TOKENIZER, "encode", mock_tokenize)
    article_tokenizer = ArticleTokenizer(
        TOKENIZER,
        MAX_SEQ_LEN,
        ext_type,
        packing_boundary=packing_boundary,
        keep_prompt_only_sequences=keep_prompt_only_sequences)
    monkeypatch.setattr(article_tokenizer, "eos_token_id", EOS_TOKEN_ID)
    monkeypatch.setattr(article_tokenizer.packer, "eos_token_id", EOS_TOKEN_ID)
    return article_tokenizer


@pytest.mark.parametrize(
    'packing_boundary,ext_type,prompt,completion,gold_token_ids,gold_ttids',
    [
        (BoundaryType.JSONL, FileExtension.JSONL, "", "", [],
         []),  # empty prompt and completion
        (BoundaryType.JSONL, FileExtension.JSONL, "", "bye", [2, 0],
         [COMP, COMP]),  # empty prompt
        (BoundaryType.JSONL, FileExtension.JSONL, "hi", "", [1, 0],
         [PROMPT, COMP]),  # empty completion
        (BoundaryType.JSONL, FileExtension.JSONL, "hi", "bye", [1, 2, 0],
         [PROMPT, COMP, COMP]),  # full prompt and completion
        (BoundaryType.JSONL, FileExtension.JSONL, "hi test", "bye test",
         [1, 3, 2, 3, 0], [PROMPT, PROMPT, COMP, COMP, COMP])
    ])
def test_tokenize(article_tokenizer: ArticleTokenizer, prompt: str,
                  completion: str, gold_token_ids: List[int],
                  gold_ttids: List[int]):
    token_ids, ttids = article_tokenizer.tokenize(completion, prompt)

    assert token_ids == gold_token_ids
    assert ttids == gold_ttids


@pytest.mark.parametrize(
    'packing_boundary,ext_type,prompt,completion,gold_prompt,gold_completion',
    [
        (BoundaryType.JSONL, FileExtension.JSONL, "hi", "bye", "hi",
         " bye"),  # no spaces
        (BoundaryType.JSONL, FileExtension.JSONL, " hi", "bye", " hi",
         " bye"),  # space in front of prompt
        (BoundaryType.JSONL, FileExtension.JSONL, "hi ", "bye", "hi",
         " bye"),  # space after prompt
        (BoundaryType.JSONL, FileExtension.JSONL, "hi", " bye", "hi",
         " bye"),  # space before completion
        (BoundaryType.JSONL, FileExtension.JSONL, "hi", "bye ", "hi",
         " bye "),  # space after completion
        (BoundaryType.JSONL, FileExtension.JSONL, "hi ", " bye", "hi",
         " bye"),  # space after prompt before completion
        (BoundaryType.JSONL, FileExtension.JSONL, " h i ", " b y e ", " h i",
         " b y e ")  # spaces in other places
    ])
def test_add_space_separator(article_tokenizer: ArticleTokenizer, prompt: str,
                             completion: str, gold_prompt: List[int],
                             gold_completion: List[int]):
    completion_test, prompt_test = article_tokenizer._add_space_separator(
        completion, prompt)

    assert completion_test == gold_completion
    assert prompt_test == gold_prompt


@pytest.mark.parametrize(
    'packing_boundary,ext_type,jsonl,gold_token_ids,gold_ttids,keep_prompt_only_sequences',
    [
        (BoundaryType.JSONL, FileExtension.JSONL, {
            "prompt": "",
            "completion": ""
        }, [], [], True),  # single jsonl empty
        (BoundaryType.JSONL, FileExtension.JSONL, {
            "prompt": "hi",
            "completion": ""
        }, [[1, 0]], [[PROMPT, SEP]], True),
        (BoundaryType.JSONL, FileExtension.JSONL, {
            "prompt": "",
            "completion": "bye"
        }, [[2, 0]], [[COMP, SEP]], True),
        (BoundaryType.JSONL, FileExtension.JSONL, {
            "prompt": "hi test",
            "completion": "bye test"
        }, [[1, 3, 2, 3, 0]], [[PROMPT, PROMPT, COMP, COMP, SEP]
                               ], True),  # single jsonl prompt and completion
        (BoundaryType.JSONL, FileExtension.JSONL, [{
            "prompt": "hi",
            "completion": "bye"
        }], [[1, 2, 0]], [[PROMPT, COMP, SEP]], True),  # list length 1
        (BoundaryType.JSONL, FileExtension.JSONL, [{
            "prompt": "hi",
            "completion": "bye"
        }, {
            "prompt": "test",
            "completion": "test"
        }], [[1, 2, 0, 3, 3, 0]], [[PROMPT, COMP, COMP, PROMPT, COMP, SEP]
                                   ], True),  # list length 2
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL, {
            "prompt": "",
            "completion": ""
        }, [], [], True),
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL, {
            "prompt": "hi",
            "completion": ""
        }, [[1, 0]], [[PROMPT, SEP]], True),
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL, {
            "prompt": "",
            "completion": "bye"
        }, [[2, 0]], [[COMP, SEP]], True),
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL, {
            "prompt": "hi test",
            "completion": "bye test"
        }, [[1, 3, 2, 3, 0]], [[PROMPT, PROMPT, COMP, COMP, SEP]], True),
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL,
         [{
             "prompt": "hi",
             "completion": "bye"
         }], [[1, 2, 0]], [[PROMPT, COMP, SEP]], True),
        (BoundaryType.PROMPT_COMPLETION_PAIR, FileExtension.JSONL,
         [{
             "prompt": "hi",
             "completion": "bye"
         }, {
             "prompt": "test",
             "completion": "test"
         }], [[1, 2, 0], [3, 3, 0]], [[PROMPT, COMP, COMP],
                                      [PROMPT, COMP, SEP]], True)
    ])
def test_process_jsonl(
        article_tokenizer_for_prompt_sequences: ArticleTokenizer,
        jsonl: Union[dict, List], gold_token_ids: List[List[int]],
        gold_ttids: List[List[int]], keep_prompt_only_sequences: bool):
    tokenized_articles = article_tokenizer_for_prompt_sequences.process_jsonl(
        jsonl)
    for tokenized_article, gold_token, gold_ttid in zip(
            tokenized_articles, gold_token_ids, gold_ttids):
        assert tokenized_article.token_ids == gold_token
        assert tokenized_article.token_type_ids == gold_ttid


@pytest.mark.parametrize(
    'packing_boundary,ext_type,text,gold_token_ids,gold_ttids',
    [
        (BoundaryType.JSONL, FileExtension.JSONL, "", [], []),  # empty text
        (BoundaryType.JSONL, FileExtension.TXT, "hi", [1, 0], [COMP, SEP
                                                               ]),  # one word
        (BoundaryType.JSONL, FileExtension.TXT, "hi bye test", [1, 2, 3, 0],
         [COMP, COMP, COMP, SEP]),  # multiple word
    ])
def test_process_text(article_tokenizer: ArticleTokenizer, text: str,
                      gold_token_ids: List[int], gold_ttids: List[int]):
    tokenized_articles = article_tokenizer.process_text(text)
    assert tokenized_articles[0].token_ids == gold_token_ids
    assert tokenized_articles[0].token_type_ids == gold_ttids


def get_tokenized_seq(token_ids: List[int],
                      token_type_ids: List[int]) -> TokenizedSequence:
    return TokenizedSequence(token_ids, token_type_ids, MAX_SEQ_LEN,
                             EOS_TOKEN_ID)


@pytest.mark.parametrize(
    'packing_boundary,ext_type,article,gold_tokenized_sequence,gold_unfinished_sequence',
    [
        (BoundaryType.JSONL, FileExtension.JSONL, None, [],
         get_tokenized_seq([], [])),  # test None passed in
        (BoundaryType.JSONL, FileExtension.TXT, "hi bye test", [],
         get_tokenized_seq([1, 2, 3, 0], [COMP, COMP, COMP, SEP])),
        (BoundaryType.JSONL, FileExtension.JSONL,
         '{"prompt": "hi", "completion": "bye"}', [],
         get_tokenized_seq([1, 2, 0], [PROMPT, COMP, SEP])),
        (BoundaryType.JSONL, FileExtension.JSONL,
         '[{"prompt": "hi", "completion": "bye"}, {"prompt": "test", "completion": "test"}, \
        {"prompt": "test", "completion": "test"}]', [
             get_tokenized_seq([1, 2, 0, 3, 3, 0],
                               [PROMPT, COMP, COMP, PROMPT, COMP, COMP])
         ], get_tokenized_seq([3, 3, 0], [PROMPT, COMP, SEP]))
    ])
def test__call__(article_tokenizer: ArticleTokenizer, article: Optional[str],
                 gold_tokenized_sequence: List[TokenizedSequence],
                 gold_unfinished_sequence: List[TokenizedSequence]):
    tokenized_sequence = article_tokenizer(article)
    assert tokenized_sequence == gold_tokenized_sequence
    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence


@pytest.mark.parametrize(
    'packing_boundary,ext_type,articles,gold_tokenized_sequences,gold_unfinished_sequence',
    [(BoundaryType.JSONL, FileExtension.TXT, ["hi bye", "hi bye", "hi hi"],
      [[], [],
       [
           get_tokenized_seq([1, 2, 0, 1, 2, 0],
                             [COMP, COMP, SEP, COMP, COMP, SEP])
       ]], get_tokenized_seq([1, 1, 0], [COMP, COMP, SEP])),
     (BoundaryType.JSONL, FileExtension.JSONL, [
         '{"prompt": "hi", "completion": "bye"}',
         '{"prompt": "hi", "completion": "bye"}', '{ \
        "prompt": "hi", "completion": "hi"}'
     ], [[], [],
         [
             get_tokenized_seq([1, 2, 0, 1, 2, 0],
                               [PROMPT, COMP, SEP, PROMPT, COMP, SEP])
         ]], get_tokenized_seq([1, 1, 0], [PROMPT, COMP, SEP]))])
def test_multiple__call__(
        article_tokenizer: ArticleTokenizer, articles: List[Optional[str]],
        gold_tokenized_sequences: List[List[TokenizedSequence]],
        gold_unfinished_sequence: List[TokenizedSequence]):
    for article, gold_tokenized_sequence in zip(articles,
                                                gold_tokenized_sequences):
        tokenized_sequence = article_tokenizer(article)
        assert tokenized_sequence == gold_tokenized_sequence

    assert article_tokenizer.packer.unfinished_sequence == gold_unfinished_sequence


@pytest.mark.parametrize(
    'packing_boundary,ext_type,articles,gold_token_ids,gold_token_type_ids,keep_prompt_only_sequences',
    [(BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi bye test hi bye", "completion": ""}]', [], [], False),
     (BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi bye test hi bye", "completion": ""}]', [[
          1, 2, 3, 1, 2, 0
      ]], [[PROMPT, PROMPT, PROMPT, PROMPT, PROMPT, SEP]], True),
     (BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi", "completion": ""}, \
      {"prompt": "hi bye", "completion": ""}, \
      {"prompt": "hi bye test", "completion": ""}]', [], [], False),
     (BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi hi hi hi hi hi hi hi", "completion": "bye"}]',
      [[1, 1, 2, 0, 0, 0]], [[PROMPT, PROMPT, COMP, SEP, PAD, PAD]], False),
     (BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi hi hi hi hi hi hi hi", "completion": "bye"}]', [
          [1, 1, 1, 1, 1, 1], [1, 1, 2, 0, 0, 0]
      ], [[PROMPT, PROMPT, PROMPT, PROMPT, PROMPT, PROMPT],
          [PROMPT, PROMPT, COMP, SEP, PAD, PAD]], True),
     (BoundaryType.JSONL, FileExtension.JSONL,
      '[{"prompt": "hi bye test hi bye test", "completion": ""}, \
            {"prompt": "hi bye test hi", "completion": "test hi"}]', [[
          1, 2, 3, 1, 3, 1
      ], [0, 0, 0, 0, 0, 0]], [[PROMPT, PROMPT, PROMPT, PROMPT, COMP, COMP],
                               [SEP, PAD, PAD, PAD, PAD, PAD]], False)])
def test_prompt_only_sequences(
        article_tokenizer_for_prompt_sequences: ArticleTokenizer,
        articles: Optional[str], gold_token_ids: List[int],
        gold_token_type_ids: List[int], keep_prompt_only_sequences: bool):
    sequences = []
    sequences += article_tokenizer_for_prompt_sequences(articles)
    sequences += article_tokenizer_for_prompt_sequences(None)
    token_ids = [seq.token_ids for seq in sequences]
    token_tids = [seq.token_type_ids for seq in sequences]
    assert token_ids == gold_token_ids
    assert token_tids == gold_token_type_ids
