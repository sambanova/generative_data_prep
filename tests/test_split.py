import tempfile

import pytest
from transformers import GPT2Config, GPT2Tokenizer

from generative_data_prep.data_prep import pipeline_main
from generative_data_prep.utils import BoundaryType, PackingConfig


def test_split_should_throw():
    """Test that an assert error should be thrown when the number of splits exceeds the number of entries."""
    file_path = "tests/examples/generative_tuning/example_generative_tuning_data.jsonl"
    TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
    MODEL_CONFIG = GPT2Config.from_pretrained("gpt2")

    with pytest.raises(Exception) as e_info:
        with tempfile.TemporaryDirectory() as output_dir:
            pipeline_main(
                input_file_path=file_path,
                tokenizer=TOKENIZER,
                model_config=MODEL_CONFIG,
                output_dir=output_dir,
                overwrite_output_path=False,
                num_training_splits=128,
                num_dev_splits=0,
                num_test_splits=0,
                max_seq_length=1024,
                disable_space_separator=False,
                keep_prompt_only_sequences=True,
                input_packing_config=PackingConfig.get_default(),
                attention_boundary=BoundaryType.JSONL,
                dev_ratio=None,
                test_ratio=None,
                packing_boundary=BoundaryType.JSONL,
                prompt_keyword="prompt",
                completion_keyword="completion",
                shuffle="False",
                num_workers=1,
                do_not_balance_hdf5=True,
                keep_split_jsonls=True,
            )

    assert (
        str(e_info.value)
        == """The number of total splits exceeds the number of
        lines in the input path jsonl file. Please reduce the number
        of splits, or increase the number of lines in the dataset."""
    )
