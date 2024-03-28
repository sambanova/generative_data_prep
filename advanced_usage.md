# Advanced Usage

Here we have some documentation on some advanced usage patterns for the generative data prep package.

## Table of contents
- [Tokenize Individual Files](#tokenize-individual-files)
    - [Individual Tokenization Flags](#individual-tokenization-flags)
- [How to use pydantic model](#how-to-use-pydantic-model)
- [How to check for corruption](#how-to-check-for-corruption)
- [Contributing](#contributing)
    - [Running Tests](#running-tests)

</br>

## Tokenize Individual Files
The `generative_data_prep/data_prep/data_prep.py` script tokenizes a single jsonline file and converts it to an HDF5 file. Training with SambaStudio requires multiple split HDF5 files, so `generative_data_prep/data_prep/pipeline.py` takes care of the splitting and processing holistically. If you have multiple split jsonline files that have been custom split and you would like to tokenize them directly, you can use the following example command and the flags list to process each file individually.

### Example
```shell
python3 -m generative_data_prep data_prep --input_file_path=<PATH TO DATASET FILE> --output_path=<PATH TO OUTPUT DIRECTORY> --pretrained_tokenizer=gpt2 --max_seq_length=1024
```


### Individual Tokenization Flags

| Flag Name  | Type | Default | Options | Description |
| --- | --- | --- | --- | --- |
| `input_file_path`  | str | REQUIRED | Any existing file path | Path to the input dataset where each line is of the form specified in [Input Format](#input-format).|
| `output_path` | str | `input_file_path`'s directory | Any valid directory path | The directory to store the output files |
| `log_file_path` | str | `output_path`'s directory/logs.log | Any valid file path | The file to save the logs in, this will save the date and time, git commit hash, input arguments and metrics associated with the dataset. |
| `overwrite_output_path` | bool | False | Include flag for True, no arguments | Permission to delete and overwrite files in `output_path`. |
| `pretrained_tokenizer` | str | None | Valid tokenizer key from Huggingface | The pretrained tokenizer to be used, loaded using transformers.AutoTokenizer.from_pretrained(args.pretrained_tokenizer), in lieu of a --tokenizer_class, --vocab_file and --merges_file. |
| `tokenizer_class` | str | 'gpt2' | ['gpt2'] | Tokenizer class to use, defaults to "gpt2" (transformers.GPT2Tokenizer). If --pretrained_tokenizer is not specified, this is required. |
| `vocab_file` | str | None | Valid file path | The vocabulary file for the tokenizer. Should be a .json file for the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` is not specified, this is required. It should be a .json file for a GPT2 tokenizer. |
| `merges_file` | str | None | Valid file path | The merges file to be used with the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` tokenizer is not specified, this is required. It should be a .txt file for a GPT2 tokenizer. |
| `special_tokens_dict` | str | None | string representation of json | Any non-standard special tokens in JSON format to add to tokenizer. e.g. \"{'sep_token': \"[SEP]\"}\". |
| `max_seq_length` | int | 2048 | 512 for gpt2 small, 1024 for gpt-xl, 2048 for gpt3-13B. | The maximum sequence length of the model you are using. |
| `input_packing_config` | PackingConfig | 'full' | ['full', 'single::truncate_left', 'single::truncate_right', 'single::drop', 'greedy::truncate_left', 'greedy::truncate_right', 'greedy::drop'] | The first argument in the packing config defines the method of placing text into sequences, the second argument defines how to handle jsonls that do not fit within the max_seq_length. 'full': Defines the entire packing config, Completely fill sequences with tokens, as soon as sequences is full start packing into new sequence. Ignore article boundaries, they may be split across multiple sequences. 'greedy': Fit as many articles as possible into a sequence, make sure no article is split across multiple sequences. Fill the left over space in each sequence with padding. 'single': Each sequence contains only 1 article.  Fill the rest of the sequence with padding.  'drop': Drop the entire article if there are any tokens that overflow beyond the max sequence length.  'truncate_left':  Truncate the article from the left if there are any tokens that overflow beyond the max sequence length.  'truncate_right':  Truncate the article from the right if there are any tokens that overflow beyond the max sequence length.|
| `packing_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | 'jsonl': When packing text into sequences, keeps json lines together. This means that for greedy or single packing if the entire line does not fit in the sequences it will be thrown out. 'prompt_completion_pair': When packing text into sequences, prompt_completion_pairs together, but may break up json lines that contain a list of prompt completion pairs. |
| `attention_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | The boundary to use when training with --article_attention flag. If you choose prompt_completion_pair tokens will only attend to tokens in the prompt_completion_pair. If you choose jsonl, then tokens will attend to all the prompt completion pairs in the jsonl |
| `prompt_keyword` | str | 'prompt' |  | If your input json has a string keyword for prompt other than "prompt", place the keyword here. e.g Input_json: {"source": ... "target": ...} -> --prompt_keyword='source'. |
| `completion_keyword` | str | 'completion' |  | If your input json has a string keyword for completion other than "completion", place the  keyword here. e.g Input_json: {"source": ... "target": ...} -> --completion_keyword='target'. |
| `prompt_prefix` | str | 'None' | | text to add before the prompt, for chatML conventions use (e.g. "\<human\>:") |
| `prompt_postfix` | str | 'None' | | text to add after the prompt, for chatML conventions use (e.g. "\<bot\>:") |
| `disable_space_separator` | bool | False | Include flag for True, no arguments |  If you include this flag, NO spaces will be prepended to the completion. (If you do not add this flag then a space is added to every completion if it does not already have a space). Including this flag is dangerous and not recommended because if you have input data like {"prompt": "hello." "completion": "how are you?"}, when the prompt and completion are combined it will look like "hello.how are you?" which will mess up the tokenization.--completion_keyword='target'. |
| `keep_prompt_only_sequences` | bool | False | Include flag for True, no arguments | If you include this flag, packed sequences with only prompt tokens will not be dropped. Data with only prompt will be dropped by default because training with prompt-only sequences with prompt_loss_weight=0.0 may lead to errors. Data is dropped because of one of the following conditions: 1. the input file data prompt completion pairs contains only a prompt. 2. If the sequence is truncated such that only prompt tokens remain |
| `categories_path` | bool | False | If you include this flag, then the 'category' field from your input jsonls will be stored in the 'category_id' dataset in your output hdf5 files. This flag must point to the file path of a json file that contains a list of all the strings of the 'category' keys in your dataset.|

</br>

## How to use pydantic model

If you want to use the output `metadata.yaml` file to validate your dataset parameters adhere to a certain schema, you can use the custom pydantic model that we provide in this library. Below is an example of how to do that:

```
import yaml
from generative_data_prep.utils import DatasetMetadata
from transformers import AutoConfig

# Load in the metadata file using pyyaml
metadata_file = os.path.join(output_dir, "metadata.yaml")
with open(metadata_file, "r") as file:
    metadata_dict = yaml.safe_load(file)

gpt2_config = AutoConfig.from_pretrained("gpt2")
training_param_dict = {
    "eval": False,
    "batch_size": 1,
    "model_type": str(type(gpt2_config)),
    "use_token_type_ids": use_token_type_ids,
    "vocab_size": gpt2_config.vocab_size,
    "number_of_workers": 4,
    "max_seq_length": 1024,
}

DatasetMetadata.model_validate(metadata_dict, context=training_param_dict)
```

If DatasetMetadata does not error out then that means the training parameters meet the requirements of the dataset!

If DatasetMetadata does error out, that means that the training parameters need to be modified or the dataset needs to be generated again to fit the needs of the training parameters. There will be a detailed error output indicating which parameters of the metadata are not compatible.

## How to check for corruption

The generative data prep package provides a utility for checking if your dataset has been corrupted using metadata recorded during processing. This metadata contains each of the different files paired with their size, modified date, and sha256 hash. If included, this verification should be used before running the pydantic model to make sure nothing is wrong with the dataset.

We also create an overall metadata file for each of the files within the output directory! This metadata file contains each of the different files paired with their size, modified date, and sha256 hash. This allows for users to check to see if their dataset has been corrupted; thus invalidating the datasetMetadata pydantic model as there could be some hidden errors. This verification should be used before running the pydantic model to make sure nothing is wrong with the dataset.

Here is an example:

```
from generative_data_prep.utils import validate_sha256

validate_sha256(output_dir)
```
`output_dir` here should point to the directory which was created using generative data pipeline. This function returns a `bool` and will be `True` if there is NO corruption in the dataset and `False` if there is corruption in the dataset.

Under the hood each file is scrubbed through and is first checked with the size and modified date. If these values are not the same as when the file was first created then the function will calculate the sha256 of the file and compare it to what is saved.

## Contributing
For those looking to contribute, please follow the [contribution guide](.github/CONTRIBUTING.rst).

### Running tests

The following commands allow you to run the test suite for this package. Make sure you've set up your virtual environment with the proper dependencies!

```
pip install .
pip install -r requirements/tests-requirements.txt
pytest
```

</br>
