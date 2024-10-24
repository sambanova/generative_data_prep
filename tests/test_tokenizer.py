from transformers import BertConfig, BertTokenizerFast, GPT2Config, GPT2TokenizerFast

from generative_data_prep.__main__ import get_tokenizer


def test_pretrained_tokenizer_gpt2():
    """Test the tokenizer function using a pretrained_tokenizer."""
    tokenizer, model_config = get_tokenizer(pretrained_tokenizer="gpt2", special_tokens_dict=None)
    assert type(tokenizer) == GPT2TokenizerFast
    assert type(model_config) == GPT2Config


def test_pretrained_tokenizer_bert_base_uncased():
    """Test the tokenizer function using a pretrained_tokenizer."""
    tokenizer, model_config = get_tokenizer(pretrained_tokenizer="bert-base-uncased", special_tokens_dict=None)
    assert type(tokenizer) == BertTokenizerFast
    assert type(model_config) == BertConfig
