import os
from unittest.mock import Mock, patch

import pytest
from transformers import PretrainedConfig

from generative_data_prep.utils import (
    get_max_seq_length_arg,
    training_to_data_prep_params,
    verify_enough_data_to_run_one_batch,
)
from tests.conftest import TESTS_EXAMPLES_PATH


def get_input_path(test_name: str) -> str:
    """Create an absolute path to an example input."""
    base_path = TESTS_EXAMPLES_PATH / test_name / f"example_{test_name}_data"
    if os.path.isdir(base_path):
        return base_path
    else:
        ext = ".txt" if "txt" in test_name else ".jsonl"
        return f"{base_path}{ext}"


# Test fixtures
@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    config = Mock(spec=PretrainedConfig)
    config.max_position_embeddings = 1024
    config.vocab_size = 50000
    return config


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.vocab_size = 50000
    return tokenizer


@pytest.fixture
def temp_input_file(tmp_path):
    """Create a temporary input file."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text for testing")
    return str(input_file)


# Test get_max_seq_length_arg remains unchanged
@pytest.mark.parametrize(
    "config_attrs,expected,raises_error",
    [
        ({"max_position_embeddings": 512}, 512, False),
        ({"n_positions": 1024}, 1024, False),
        ({}, None, True),
    ],
)
def test_get_max_seq_length_arg(config_attrs, expected, raises_error):
    """Test getting max sequence length from different config types."""
    config = Mock()

    # Clear any default attributes
    config.max_position_embeddings = None
    config.n_positions = None

    # Set only the specified attributes
    for attr, value in config_attrs.items():
        setattr(config, attr, value)
    if raises_error:
        with pytest.raises(ValueError):
            result = get_max_seq_length_arg(config)
    else:
        result = get_max_seq_length_arg(config)
        assert str(result) == str(expected)


# Test verify_enough_data_to_run_one_batch with temporary file
@pytest.mark.parametrize(
    "file_size,num_splits,grad_steps,batch_size,seq_length,num_rdus,expected",
    [
        (1000000, 4, 1, 8, 512, 2, 4),  # Enough data
        (100, 8, 1, 8, 512, 2, None),  # Not enough data, should raise ValueError
    ],
)
def test_verify_enough_data_to_run_one_batch(
    tmp_path, file_size, num_splits, grad_steps, batch_size, seq_length, num_rdus, expected
):
    """Test verification of sufficient data for batch processing."""
    # Create temporary file with specified size
    input_file = tmp_path / "test_input.txt"
    input_file.write_bytes(b"x" * file_size)

    if file_size < (num_splits * grad_steps * batch_size * seq_length * 3):
        with pytest.raises(ValueError):
            verify_enough_data_to_run_one_batch(
                str(input_file), num_splits, grad_steps, batch_size, seq_length, num_rdus
            )
    else:
        result = verify_enough_data_to_run_one_batch(
            str(input_file), num_splits, grad_steps, batch_size, seq_length, num_rdus
        )
        assert result == expected


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoConfig.from_pretrained")
@pytest.mark.parametrize(
    "test_scenario",  # Changed from input_path to test_scenario
    [
        "generative_tuning",  # Equal sizes
        "generative_tuning",  # Tokenizer smaller than model
        "generative_tuning",  # Tokenizer larger than model
    ],
)
def test_training_to_data_prep_params(
    mock_auto_config, mock_auto_tokenizer, mock_tokenizer, mock_model_config, tmp_path, test_scenario
):
    """Test conversion of training parameters to data preparation parameters."""
    # Setup mocks
    mock_auto_config.return_value = mock_model_config
    mock_auto_tokenizer.return_value = mock_tokenizer

    # Create temporary files
    input_file_path = get_input_path(test_scenario)  # Changed to use test_scenario
    log_file = tmp_path / "log.txt"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"

    # Create the files and directories
    log_file.touch()
    output_dir.mkdir()
    checkpoint_dir.mkdir()

    args = training_to_data_prep_params(
        input_path=str(input_file_path),
        output_path=str(output_dir),
        log_file_path=str(log_file),
        checkpoint_path=str(checkpoint_dir),
        number_of_rdus=1,
        grad_accum_steps=1,
        pef_batch_size=1,
    )

    assert args.input_path == str(input_file_path)
    assert args.output_path == str(output_dir)
    assert str(args.input_packing_config) == "greedy::drop"


@pytest.mark.parametrize(
    "input_path,tokenizer_vocab_size,model_vocab_size,should_raise",
    [
        ("generative_tuning", 50000, 50000, True),  # Equal sizes
        ("generative_tuning", 40000, 50000, False),  # Tokenizer smaller than model
        ("generative_tuning", 60000, 50000, True),  # Tokenizer larger than model
    ],
)
def test_training_to_data_prep_params_vocab_size_validation(
    input_path, tokenizer_vocab_size, model_vocab_size, should_raise, tmp_path
):
    """Test vocabulary size validation in training parameter conversion."""
    # Create fresh mocks for each test case
    mock_tokenizer = Mock()
    mock_tokenizer.vocab_size = tokenizer_vocab_size

    mock_model_config = Mock()
    mock_model_config.vocab_size = model_vocab_size
    mock_model_config.max_position_embeddings = 1024

    # Create temporary files
    input_file_path = get_input_path(input_path)
    log_file = tmp_path / "log.txt"
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"

    # Create the files and directories
    log_file.touch()
    output_dir.mkdir()
    checkpoint_dir.mkdir()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), patch(
        "transformers.AutoConfig.from_pretrained", return_value=mock_model_config
    ):
        if should_raise:
            with pytest.raises(ValueError):
                training_to_data_prep_params(
                    input_path=str(input_file_path),
                    output_path=str(output_dir),
                    log_file_path=str(log_file),
                    checkpoint_path=str(checkpoint_dir),
                    number_of_rdus=4,
                    grad_accum_steps=1,
                    pef_batch_size=8,
                )
        else:
            args = training_to_data_prep_params(
                input_path=str(input_file_path),
                output_path=str(output_dir),
                log_file_path=str(log_file),
                checkpoint_path=str(checkpoint_dir),
                number_of_rdus=1,
                grad_accum_steps=1,
                pef_batch_size=1,
            )
            assert args is not None
