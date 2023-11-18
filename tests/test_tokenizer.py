import os
import pathlib

from transformers import (
    BertConfig,
    BertTokenizerFast,
    GPT2Config,
    GPT2Tokenizer,
    GPT2TokenizerFast,
)

from generative_data_prep.__main__ import get_tokenizer


def test_pretrained_tokenizer_gpt2():
    """Test the tokenizer function using a pretrained_tokenizer."""
    tokenizer, model_config = get_tokenizer(
        pretrained_tokenizer="gpt2", tokenizer_class=None, vocab_file=None, merges_file=None, special_tokens_dict=None
    )
    assert type(tokenizer) == GPT2TokenizerFast
    assert type(model_config) == GPT2Config


def test_pretrained_tokenizer_bert_base_uncased():
    """Test the tokenizer function using a pretrained_tokenizer."""
    tokenizer, model_config = get_tokenizer(
        pretrained_tokenizer="bert-base-uncased",
        tokenizer_class=None,
        vocab_file=None,
        merges_file=None,
        special_tokens_dict=None,
    )
    assert type(tokenizer) == BertTokenizerFast
    assert type(model_config) == BertConfig


def test_tokenizer_class():
    """Test the tokenizer function using a tokenizer_class with vocab and merge files."""
    current_dir = pathlib.Path(__file__).parent.resolve()
    vocab_file = os.path.join(current_dir, "gpt2_vocab_and_merge_files", "vocab.json")
    merge_file = os.path.join(current_dir, "gpt2_vocab_and_merge_files", "merges.txt")
    tokenizer, model_config = get_tokenizer(
        pretrained_tokenizer=None,
        tokenizer_class="gpt2",
        vocab_file=vocab_file,
        merges_file=merge_file,
        special_tokens_dict=None,
    )
    assert type(tokenizer) == GPT2Tokenizer
    assert type(model_config) == GPT2Config


def test_tokenizer_class_failure():
    """Test the tokenizer function using a tokenizer_class but with the
    intention of failing due to missing vocab and merge files.
    """
    try:
        _, _ = get_tokenizer(
            pretrained_tokenizer=None,
            tokenizer_class="gpt2",
            vocab_file=None,
            merges_file=None,
            special_tokens_dict=None,
        )
    except ValueError:
        assert True
        return
    assert False


def test_tokenizer_class_not_implemented():
    """Test the tokenizer function using a tokenizer_class but with the intention of
    failing due to using a model type that isn't added to the TOKENIZER_CLASS.
    """
    current_dir = pathlib.Path(__file__).parent.resolve()
    vocab_file = os.path.join(current_dir, "gpt2_vocab_and_merge_files", "vocab.json")
    merge_file = os.path.join(current_dir, "gpt2_vocab_and_merge_files", "merges.txt")
    try:
        _, _ = get_tokenizer(
            pretrained_tokenizer=None,
            tokenizer_class="bloom",
            vocab_file=vocab_file,
            merges_file=merge_file,
            special_tokens_dict=None,
        )
    except NotImplementedError:
        assert True
        return
    assert False
