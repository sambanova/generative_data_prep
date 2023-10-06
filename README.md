[![CircleCI](https://dl.circleci.com/status-badge/img/gh/sambanova/generative_data_prep/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/sambanova/generative_data_prep/tree/main)
[![Python](https://img.shields.io/badge/python-%3E=3.7-blue.svg)](https://www.python.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![flake8](https://img.shields.io/badge/pep8-flake8-blue.svg)](https://github.com/PyCQA/flake8)
[![bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![mypy](https://img.shields.io/badge/mypy-checked-green.svg)](http://mypy-lang.org/)

<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/img/SambaNova-light-logo-1.png" height="60">
  <img alt="Text changing depending on mode. Light: 'So light!' Dark: 'So dark!'" src="/img/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

# Generative data preparation
This software package is designed for preparing data that can be used to train generative models. It offers an efficient way to convert input text files into tokenized sequences that are packed into a fixed sequence length. The resulting output directory can be directly used for training with SambaStudio. This package features many styles of packing text of any length into tokenized sequences, compressed HDF5 file outputs, efficient multiprocessing, shuffling any sized dataset, splitting your data into train/dev/test, and specifying what tokens are attended to during training.

## Table of contents
- [Contributing](#Contributing)
- [Installation](#installation)
- [Requirements](#requirements)
- [Introduction](#introduction)
- [Input format](#input-format)
- [End to end data preparation](#end-to-end-data-preparation)
    - [Input Flags](#flags)
- [Tokenizing one file](#tokenizing-one-file)
    - [Input Flags](#tokenize-one-file-flags)
- [Running tests](#running-tests)
- [Example use cases](#example-use-cases)
    - [Pretraining](#pretraining)
    - [Generative tuning](#generative-tuning)
    - [Dialogue](#dialogue)
    - [Meta in context learning](#meta-in-context-learning)

## Contributing
Please follow the [contribution guide](.github/CONTRIBUTING.rst).

## Installation
```
git clone https://github.com/sambanova/generative_data_prep.git
cd generative_data_prep
pip install .
```

## Requirements
- Python version 3.7+, **only verified on python 3.7.6**
- Support for linux and mac OS. Not tested on Windows

## Introduction
The `generative_data_prep/data_prep/pipeline.py` script is designed to facilitate end-to-end data preparation for training machine learning models. This script takes a single [jsonline](https://jsonlines.org/) or text file as input, shuffles it, splits it into multiple train/dev/test files, then calls `generative_data_prep/data_prep/data_prep.py` on all the splits to tokenize the text, pack into fixed length sequences and convert to [HDF5 format](https://www.geeksforgeeks.org/hdf5-files-in-python/). The directory path passed in to this repo as `output_path` can be used directly as a training dataset.

The `generative_data_prep/data_prep/data_prep.py` script is used for tokenizing a single [jsonline](https://jsonlines.org/) or text file, packing it into fixed length sequences and converting it to [HDF5 format](https://www.geeksforgeeks.org/hdf5-files-in-python/).  However, when training with SambaStudio, multiple split HDF5 files are needed to run data parallel training. Therefore, unless you already have multiple split input files that you want to tokenize directly, we recommend using the `pipeline.py` script for end-to-end data preparation.

## Input format

Each line in the input file must be either plain text or [jsonline](https://jsonlines.org/). if the jsonline has different keywords, refer to the `prompt_keyword`, and `completion_keyword` flag documentation below.

Each line in the input file can be formatted as one of the following:
- Plain text
- `{"prompt": "", "completion": ""}`
- `[{"prompt": "text...", "completion": "text..."}, {"prompt": "text...", "completion": "text..."}, {"prompt": "text...", "completion": "text..."}, ...]`


## End to end data preparation
The `generative_data_prep/data_prep/pipeline.py` script takes a single jsonline or text file as input, shuffles it, splits it into multiple train/dev/test, tokenizes the text, packs it into fixed sequence lengths and then converts it to HDF5 file format.

### Example
```python
python3 -m generative_data_prep pipeline --input_file_path=path_to_jsonl.jsonl --output_path=path_to_output_directory --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM
```

### Output
The `output_path` will contain all the tokenized HDF5 split files, and a directory called `tokenizer`. The directory path that is passed in as the `output_path` flag is where the final dataset is saved that can be used as input data to upload and run training. The `tokenizer` directory will be transferred to any output checkpoints that are saved by Sambastudio so the tokenizer can be used for inference. If you include the `keep_split_jsonls` flag, then the `output_path` will additionally contain a `splits` directory that saves the jsonl versions of the HDF5 files, meaning that splits/train_1_of_X.jsonl is the jsonl text version of train_1_of_X.hdf5.

The output HDF5 files each contain two datasets:
- \"input_ids\": sequences of tokens ids
- \"token_type_ids\": describe the type of each token. The default id assignments are:
  - id=0 for tokens in the prompt
  - id=1 for tokens in the completion
  - id=2 for \<eos\> tokens that serve as padding tokens (will not be trained to predict)
  - id=3 for \<eos\> tokens at the end of articles, that define the attention boundary when training with article attention

### Holdout Evaluation Data
To evaluate on a holdout set of data during training, pipeline.py can create splits of holdout evaluation data.

To do this, include flags from only one of the two options below, only use one option or the other. Please review the [Flags](#flags) section for in detail descriptions of these flags.
- To specify the number of training splits and evaluation splits directly, use the three flags `--num_training_splits=...`, `--num_dev_splits=...` and `--num_test_splits=...`
- To specify the percentage of the data heldout for evaluation, you can specify `--dev_ratio=0.1` and `--test_ratio=...`, where 0.1 means that approximately 10% of the data will be included in the evaluation splits. You can also specify the `--num_training_splits=...` flag to control the total number of training splits, but we recommend to let this default.

All this evaluation data will saved under the `output_path`, if you want to run evaluation on the eval_splits during training you must enable `do_eval` on SambaStudio.

### Holdout Test Data
To create a holdout set of test data that is not tokenized, pipeline.py can create these splits and will leave the data un-tokenized and save it in the `output_path/test` directory. This data is left in jsonl text format because running evaluation or inference usually requires text inputs instead of tokenized inputs.

To do this, include flags from only one of the two options below, only use one option or the other. Please review the [Flags](#tokenize-one-file-flags) section for in detail descriptions of these flags.
- To specify the number of training splits and test splits directly, use the three flags `--num_training_splits=...`, `--num_dev_splits=...` and `--num_test_splits=...`
- To specify the percentage of the data heldout for testing, you can specify `--dev_ratio=...` and `--test_ratio=0.1`, where 0.1 means that approximately 10% of the data will be included in the test splits. You can also specify the `--num_training_splits=...` flag to control the total number of training splits, but we recommend to let this default.

### Flags
<details>
  <summary>CLICK HERE to see flags</summary>

| Flag Name  | Type | Default | Options | Description |
| --- | --- | --- | --- | --- |
| `input_file_path`  | str | REQUIRED | Any existing file path | Path to the input dataset which must be in jsonline format, where each line is of the form specified in [Input Format](#input-format).|
| `output_path` | str | `input_file_path`'s directory | Any valid directory path | The directory to store the output files |
| `log_file_path` | str | `output_path`/logs.log | Any valid file path | The file to save the logs in, this will save the date and time, git commit hash, input arguments and metrics associated with the dataset. |
| `overwrite_output_path` | bool | False | Include flag for True, no arguments | Permission to delete and overwrite files in `output_path`. |
| `pretrained_tokenizer` | str | None | Valid tokenizer key from Huggingface | The pretrained tokenizer to be used, loaded using transformers.AutoTokenizer.from_pretrained(args.pretrained_tokenizer), in lieu of a `tokenizer_class`, `vocab_file` and `merges_file`. |
| `tokenizer_class` | str | 'gpt2' | ['gpt2'] | Tokenizer class to use, defaults to "gpt2" (transformers.GPT2Tokenizer). If `pretrained_tokenizer` is not specified, this is required. |
| `vocab_file` | str | None | Valid file path | The vocabulary file for the tokenizer. Should be a .json file for the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` is not specified, this is required. It should be a .json file for a GPT2 tokenizer. |
| `merges_file` | str | None | Valid file path | The merges file to be used with the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` tokenizer is not specified, this is required. It should be a .txt file for a GPT2 tokenizer. |
| `special_tokens_dict` | str | None | string representation of json | Any non-standard special tokens in JSON format to add to tokenizer. e.g. \"{'sep_token': \"[SEP]\"}\". Additional tokens can be also added using the "additional_special_tokens" keyword. For example, indentation encoding can be added with \"{'additional_special_tokens': [\"\t\", \"\t\t\", \"\t\t\t\"]}\". |
| `max_seq_length` | int | 2048 | 512 for gpt2 small, 1024 for gpt-xl, 2048 for gpt3-13B. | The maximum sequence length of the model you are using. |
| `input_packing_config` | PackingConfig | 'full' | ['full', 'single::truncate_left', 'single::truncate_right', 'single::drop', 'greedy::truncate_left', 'greedy::truncate_right', 'greedy::drop'] | The first argument in the packing config defines the method of placing text into sequences, the second argument defines how to handle jsonls that do not fit within the max_seq_length. 'full': Defines the entire packing config, Completely fill sequences with tokens, as soon as sequences is full start packing into new sequence. Ignore article boundaries, they may be split across multiple sequences. 'greedy': Fit as many articles as possible into a sequence, make sure no article is split across multiple sequences. Fill the left over space in each sequence with padding. 'single': Each sequence contains only 1 article.  Fill the rest of the sequence with padding.  'drop': Drop the entire article if there are any tokens that overflow beyond the max sequence length.  'truncate_left':  Truncate the article from the left if there are any tokens that overflow beyond the max sequence length.  'truncate_right':  Truncate the article from the right if there are any tokens that overflow beyond the max sequence length. |
| `packing_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | 'jsonl': When packing text into sequences, keeps json lines together. This means that for greedy or single packing if the entire line does not fit in the sequences it will be thrown out. 'prompt_completion_pair': When packing text into sequences, prompt_completion_pairs together, but may break up json lines that contain a list of prompt completion pairs. |
| `attention_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | The boundary to use when training with --article_attention flag. If you choose prompt_completion_pair tokens will only attend to tokens in the prompt_completion_pair. If you choose jsonl, then tokens will attend to all the prompt completion pairs in the jsonl |
| `prompt_keyword` | str | 'prompt' | | If your input json has a string keyword for prompt other than "prompt", place the keyword here. e.g Input_json: {"source": ... "target": ...} ->`prompt_keyword`='source'. |
| `completion_keyword` | str | 'completion' | | If your input json has a string keyword for completion other than "completion", place the  keyword here. e.g Input_json: {"source": ... "target": ...} -> --completion_keyword='target'. |
| `prompt_prefix` | str | 'None' | | text to add before the prompt, for chatML conventions use (e.g. "\<human\>:") |
| `prompt_postfix` | str | 'None' | | text to add after the prompt, for chatML conventions use (e.g. "\<bot\>:") |
| `disable_space_separator` | bool | False | Include flag for True, no arguments |  If you include this flag, NO spaces will be prepended to the completion. (If you do not add this flag then a space is added to every completion if it does not already have a space). Including this flag is dangerous and not recommended because if you have input data like {"prompt": "hello." "completion": "how are you?"}, when the prompt and completion are combined it will look like "hello.how are you?" which will mess up the tokenization.--completion_keyword='target'. |
| `keep_prompt_only_sequences` | bool | False | Include flag for True, no arguments | If you include this flag, packed sequences with only prompt tokens will not be dropped. Data with only prompt will be dropped by default because training with prompt-only sequences with prompt_loss_weight=0.0 may lead to errors. Data is dropped because of one of the following conditions: 1. the input file data prompt completion pairs contains only a prompt. 2. If the sequence is truncated such that only prompt tokens remain |
| `categories_path` | str | False | If you include this flag, then the 'category' field from your input jsonls will be stored in the 'category_id' dataset in your output hdf5 files. This flag must point to the file path of a json file that contains a list of all the strings of the 'category' keys in your dataset.|
| `shuffle` | str | 'False' | ['False', 'on_RAM', 'large_file'] | Choose the on_RAM option if your file is small enough to fit on RAM (If you are not sure if it fits on RAM, you can probably use this flag). If you are running a linux operating system and your file is too large to fit on RAM, please choose large_file option, this will run approximate file shuffling that can handle files of any size. If you want to do large file shuffling but you are not on linux, please shuffle the file before using this script. If the input file should not be shuffled, do not include this flag, it defaults to False. |
| `num_training_splits` | int | 32 if input_file_size < 10GB, 128 if 10GB < input_file_size <100GB, 256 if 100GB < input_file_size | | The number of training files to split input data into. We recommend you do not include this flag and allow it to default. If you do not default this flag, you have two options. Option 1: specify this flag with the `dev_ratio` and `test_ratio` flags, The total number of splits will be (`num_training_splits` / (1-`dev_ratio`-`test_ratio`)), and the number of dev and test splits are calculated accordingly. Option 2: specify this flag with the `num_dev_splits` and `num_test_splits` flags which define the number of splits directly. NOTE: the number of training splits must be greater than the number of training workers you have, and we recommend that the number of splits is a multiple of the number of workers you have. |
| `dev_ratio` | float | 0.0 | [0 - 1] | The ratio of data that should be excluded from train set and used for evaluation, defaults to 0%. If you specify this flag, do not specify `num_dev_splits` or `num_test_splits`. |
| `test_ratio` | float | 0.0 | [0 - 1] | The ratio of data that should be excluded from train set and is saved for testing. This data is not tokenized and left in jsonline format, defaults to 0%. If you specify this flag, do not specify `num_dev_splits` or `num_test_splits`. |
| `num_dev_splits` | int | None | Any int | number of dev (eval) splits. If you do not specify `dev_ratio`, you may specify this flag. If you include this flag, you must also include the `num_test_splits` and `num_training_splits` flags. |
| `num_test_splits` | int | None | Any int | Number of test splits. If you do not specify `test_ratio`, you may specify num_test_splits. If you include this flag, you must also include the `num_dev_splits` and `num_training_splits` flags. |
| `do_not_balance_hdf5` | bool | False | Include flag for True, no arguments | Include this flag if you DO NOT want to balance HDF5 files, this is not recommended unless the you are dealing with a huge amount of data (many terra bytes), or do not want shuffling between splits. |
| `keep_split_jsonls` | bool | False | Include flag for True, no arguments | If you DO NOT want to delete split jsonls files that are in text format in the `output_path/splits` directory include this flag. The only reason you would include this flag is if you want to see what text is in each HDF5, meaning that splits/train_1_of_X.jsonl is the jsonl text version of train_1_of_X.hdf5. Including this flag will increase the storage space of your dataset by more than two times. |
| `num_workers` | int | False | 0 <= `num_workers`<= # of available CPUs | The number of CPU workers to run tokenization with, if the previous run failed due to OOM, you need to decrease this number. |
</details>


## Tokenizing one file
The `generative_data_prep/data_prep/data_prep.py` script tokenizes a single jsonline file and converts it to an HDF5 file. However, training with SambaStudio requires multiple split HDF5 files. So, unless you already have multiple split jsonline files that you want to tokenize directly, we recommend using the `generative_data_prep/data_prep/pipeline.py` script.

### Example
```python
python3 -m generative_data_prep data_prep --input_file_path=path_to_jsonl.jsonl --output_path=path_to_output_file --pretrained_tokenizer=gpt2 --max_seq_length=1024
```

### Output
Each HDF5 file contains two datasets:
- \"input_ids\": sequences of token ids
- \"token_type_ids\": describe the type of each token. The id assignments are:
  - id=0 for tokens in the prompt
  - id=1 for tokens in the completion
  - id=2 for \<eos\> tokens that serve as padding tokens
  - id=3 for \<eos\> tokens at the end of articles, that serve as separators

### Tokenize One File Flags
<details>
  <summary>CLICK HERE to see flags</summary>


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
</details>

## Running tests
```
pip install ".[tests]"
pytest
```

## View decoded HDF5 files in human readable text format
```python
python3 generative_data_prep/utils/decode_hdf5.py --hdf5_file_path=path_to_hdf5_file --output_decoded_file_path=path_to_output_txt_file
```

## Track Dataset Metrics
The metrics associated with this dataset will be printed in the terminal. These metrics give some insight into how the data was packed into sequences, and information about the training dataset.
| Metric Name      | Definition | How to Interpret? |
| --------- | --------- | --------- |
| Articles | The number of lines in the input dataset. | How many text documents in the input dataset. |
| Dataset Tokens | Number of tokens in the output hdf5 dataset. | How many tokens are in the training dataset. But this includes both prompt tokens and padding tokens, so this metric does not necessarily show how many tokens will learned by the model. |
| Prompt Tokens | Number of prompt tokens in the output hdf5 dataset. | <- |
| Completion Tokens | Number of completion tokens in the output hdf5 dataset. | <- |
| Padding Tokens | Number of padding tokens in the output hdf5 dataset. | <- |
| Average Completion Length | Number of completion tokens divided by number of input articles. | How long the average completion is the dataset. |
| Average Prompt Length | Number of prompt tokens divided by number of input articles. |  How long the average prompt is the dataset. |
| Data Utilization | Percent of non-padding tokens in output HDF5 dataset divided by number of tokens in input dataset. | This metric reveals how much of the input data makes it to the output dataset. If this percent is much less than 100%, that means a lot of the input data will not be trained on. Refer to the "Dropped From Packing" or "Dropped From All Prompt" metrics to see why this is happening. |
| Dropped From Packing  | Number of tokens dropped during packing, divided by number of tokens in input dataset. | The percent of tokens are dropped because they do not fit into the sequence length, and the `input_packing_config` does not allow them to be overflowed.|
| Dropped From All Prompt | Number of tokens dropped because all the tokens in a sequence are prompt tokens, divided by the number of tokens in input dataset. | Sequences that are all prompts or padding (no completion tokens) are dropped. This is because the model will not learn anything from these sequences and the loss will be 0, which may cause errors. |
| Sequence Utilization | Average number of non-padding tokens in a sequence divided by sequence length. | The percent of the tokens in each sequence are actually used for training. This number imrpoved be changed by using different `input_packing_config`. The packing styles from highest sequence utilization to lowest are: `full`, `greedy::truncate_left` (or truncate_right), `greedy::drop`, `single::truncate_left` (or truncate_right), `single::drop`.|
| Seq Completion Utilization | Average number of completions tokens in a sequence divided by sequence length. | The percent of the tokens in a sequence are learned.|


## Example use cases
### Pretraining
Pretraining on unstructured data enables large languages models to learn general language patterns and structures that are useful for a wide range of downstream tasks. In order to prepare pretraining data, you need a large amount of unstructured text data. To prepare pretraining data use the flag `--input_packing_config=full`.

#### Example data
For pretraining you can have your data in two formats.

> [text separated by newlines.](tests/examples/pretraining_txt/example_pretraining_txt_data.txt)

> [jsonlines with empty prompts and text in the completions.](tests/examples/pretraining/example_pretraining_data.jsonl)

We recommend to use jsonlines with empty prompts and all the text in the completion, this is so that newlines in the text do not separate semantically related articles.
#### Example command

```
python3 -m generative_data_prep pipeline --input_file_path=./tests/examples/pretraining/example_pretraining_data.jsonl --output_path=./tests/examples/pretraining/pipelined_pretraining --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM --input_packing_config=full
```

> [View decoded output](tests/examples/pretraining/decoded_data_prepped_pretraining.txt)


### Generative tuning
Generative tuning or "fine tuning" is a technique used to adapt a pre-trained language model to perform better at a specific task. This approach typically involves training the model on input data that is structured as a "prompt" followed by a "completion". The prompt represents the input for a specific task, while the completion is the output that the model should generate. During training, the model learns to generate the relevant completion tokens based on the context provided by the prompt tokens.

The benefit of using this training format is that the model can learn to generate high-quality outputs for a specific task without requiring a large amount of task-specific training data. By leveraging the pre-trained language model's knowledge gained from being trained on a large corpus of text data, the fine-tuned model can quickly adapt to the new task and generate high-quality outputs with minimal training data.

When training on this kind of data using SambaStudio, set `prompt_loss_weight=0.0`. This ensures that the model does not learn to generate the prompt tokens, and only learns to generated completion tokens.

#### Example data
> [Jsonlines with a prompt and completion](tests/examples/generative_tuning/example_generative_tuning_data.jsonl)

#### Example command

```python
python3 -m generative_data_prep pipeline --input_file_path=./tests/examples/generative_tuning/example_generative_tuning_data.jsonl --output_path=./tests/examples/generative_tuning/pipelined_generative_tuning --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM --input_packing_config=single::drop
```

> [View decoded output](tests/examples/generative_tuning/decoded_data_prepped_generative_tuning.txt)

### Dialogue
Dialogue data often involves multiple turns in a conversation between a user and an agent. In order to train on this data, the entire conversation needs to be in the same sequence of tokens and the model should only learn to generate the agents responses based on the users inputs. To prepare data like this create a list of prompt completion pairs, and if you train with `packing_boundary=jsonl` and `input_packing_config=greedy::truncate_right/` or `input_packing_config=single::truncate_right` then these conversations are guaranteed to be in the provided order in the same sequence. Additionally if you include the `prompt_loss_weight=0.0` option while training on SambaStudio, only the completions will be learned. Also for training dialogue in chat-ml style, users can set `prompt_prefix` and `prompt_postfix`.

#### Example data
> [Lists of prompt completion pairs that represent turns in a conversation](tests/examples/dialogue/example_dialogue_data.jsonl)

#### Example command

```python
python3 -m generative_data_prep pipeline --input_file_path=./tests/examples/dialogue/example_dialogue_data.jsonl --output_path=./tests/examples/dialogue/pipelined_dialogue --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM --input_packing_config=single::truncate_right
```

> [View decoded output](tests/examples/dialogue/decoded_data_prepped_dialogue.txt)

### Meta in context learning
[Meta In Context Learning](https://arxiv.org/pdf/2110.15943.pdf) improves the few shot performance of a model by including training data formatted in a few shot style. This infrastructure allows you to prepare data in a variant of meta in context learning SambaNova uses called "All Shot" learning. In order to prepare data in this format prepare lists of prompt completion pairs, where every list contains prompt completion pairs that are completing the same instruction/task. Then prepare the data with the `input_packing_config=greedy::drop`, `packing_boundary=prompt_completion_pair` and `attention_boundary=jsonl`. This ensures that every sequence contains prompt completion pairs following the same "instruction", and that when learning a completion the model is attending to all the other prompt completion pairs before it.

#### Example data
> [Lists of prompt completion pairs that are all from the same task](tests/examples/metaICL/example_metaICL_data.jsonl)

#### Example command

```python
python3 -m generative_data_prep pipeline --input_file_path=./tests/examples/metaICL/example_metaICL_data.jsonl --output_path=./tests/examples/metaICL/pipelined_metaICL --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM --input_packing_config=greedy::drop --packing_boundary=prompt_completion_pair --attention_boundary=jsonl
```

> [View decoded output](tests/examples/metaICL/decoded_data_prepped_metaICL.txt)
