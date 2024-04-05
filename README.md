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

This software package is a flexible and efficient tool that sports features like efficient multiprocessing, shuffling data that outsizes RAM, and specifying tokens to attend to during training. By using this package, you can prepare datasets that will be used to train generative models on SambaStudio.

The [`pipeline.py`](https://github.com/sambanova/generative_data_prep/blob/main/generative_data_prep/data_prep/pipeline.py) script streamlines the data preparation process. It takes a single input file, shuffles and splits it into train/dev/test files, tokenizes, sequences, and converts them to HDF5 format using the utilities in [`data_prep.py`](https://github.com/sambanova/generative_data_prep/blob/main/generative_data_prep/data_prep/data_prep.py). The output directory contains multiple split HDF5 files that are needed to run data parallel training. This output directory will be directly used as a training dataset in SambaStudio. While this package features simple flows that work out of the box, it also supports more customization allowing for many styles of packing varied length text into tokenized sequences.

If you are an advanced user looking to process data with pre-defined splits, integrate with the package validation tools, or contribute, check out the [Advanced Usage](#advanced-usage) section below!

</br>

## Table of contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Input](#input)
- [Output](#output)
- [Flags](#flags)
- [Examples](#examples)
    - [Pretraining](#pretraining)
    - [Fine-tuning](#fine-tuning)
    - [Dialogue](#dialogue)
    - [Meta in context learning](#meta-in-context-learning)
- [Understanding Command Outputs](#understanding-outputs)
- [Advanced Usage](#advanced-usage)

</br>

## Requirements
- Python version 3.9+, **only verified on python 3.9**
- Support for Linux and Mac OS. Not tested on Windows

</br>

## Installation
```
git clone https://github.com/sambanova/generative_data_prep.git
cd generative_data_prep
pip install .
```

</br>

## Getting Started

The following simple example will help you get started with your first processed dataset:

### Example
```shell
python3 -m generative_data_prep pipeline --input_file_path=<PATH TO DATASET FILE> --output_path=<PATH TO OUTPUT DIRECTORY> --pretrained_tokenizer=gpt2 --max_seq_length=1024 --input_packing_config='greedy::drop' --shuffle=on_RAM
```

Here are a few important parameters to know about when running this example:

|          Flag Name          | Type | Description &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Instructions |
|            ---              | ---  |     ---     |     ---      |
|   `input_file_path`         |  str | An existing file path to the dataset to be processed. File must be in `.jsonl` or `.txt` format.   |  Check out the [input](#input) section for more details.    |
|   `output_path`             |  str | A path to the desired output location for the directory of processed dataset files. If the path doesn't exist, a new directory will be created using the provided path.   |   Processed datasets consist of multiple files under an output directory. If I want my output directory to be named `out`, I could put the path `/Users/johndoe/Documents/datasets/dataset_name/out` for example. Check out the [output](#output) section for more details. |
|   `pretrained_tokenizer`    |  str | The tokenizer to use when tokenizing the input dataset. The tokenizers are model specific and can be found on HuggingFace.      |  You can use the model ID from the HuggingFace model card. I.e. For Mistral-7B-v0.1 I would put `"mistralai/Mistral-7B-v0.1"`  |
|   `max_seq_length`          |  int | The max size of input sequence a model can support. This is model specific - i.e. for `GPT-2` it's __1024__, for `Llama-2` it's __4096__.  |   You can find this information in a few places, but you can often find the models max sequence length by looking in the `config.json` file under the HuffingFace model's _"File's and Versions"_ tab and finding the value of `max_position_embeddings`.  |
|   `input_packing_config`    |  str | Defines the strategy used to pack the provided data into sequences across the output HDF5 files. |   There are 7 options for this flag: `'full'`, `'single::truncate_left'`, `'single::truncate_right'`, `'single::drop'`, `'greedy::truncate_left'`, `'greedy::truncate_right'`, `'greedy::drop'`. Check out the [`input_packing_config`](#input_packing_config) flag below for an in depth description of these options. |
|          `shuffle`          |  int | Determines whether to shuffle the input dataset, and whether to shuffle on RAM. |   There are 3 options for this flag: `'False'`, `'on_RAM'`, `'large_file'`. Check out the [`shuffle`](#shuffle) flag below for more details.  |


</br>

## Input

The input file format must be either `.txt` or [`.jsonl`](https://jsonlines.org/).

### `.jsonl` Format

The JSON Lines format can be used for [fine-tuning](#fine-tuning), or [pretraining](#pretraining)/continual pre-training. Each line in the `.jsonl` format should be a json object with a `prompt`, and `completion` element. For example:

```
{"prompt": "What did the fox do?", "completion": "The quick brown fox jumped over the lazy dog."}
{"prompt": "How much wood does a woodchuck chuck?", "completion": "A woodchuck chucks 1000 wood."}
{"prompt": "Who sells seashells by the sea shore?", "completion": "She sells seashells by the sea shore."}
```

We also support lists of prompt/completion pairs within a `.jsonl` file. This will guarantee that the pairs in the list won't be shuffled apart. In other words, all promp/completion pairs on separate lines get shuffled. Here's an example structure:

```
{"prompt": "What did the fox do?", "completion": "The quick brown fox jumped over the lazy dog."}
[{"prompt": "How much wood does a woodchuck chuck?", "completion": "A woodchuck chucks 1000 wood."}, {"prompt": "How much wood can a woodchuck not chuck?", "completion": "A woodchuck cannot chuck more than 5000 wood."}]
{"prompt": "Who sells seashells by the sea shore?", "completion": "She sells seashells by the sea shore."}
```


If the JSON objects in your `.jsonl` contain keywords other than **prompt** and **completion**, refer to the `prompt_keyword` and `completion_keyword` flags [below](#prompt_keyword)

### `.txt` Format

This format should be used for pre-training/continual pretraining, but not fine-tuning. With text files, all sequences are used as completions, so all processed sequences end up having empty prompts in the prompt/completion pair. For example:

```txt
The quick brown fox jumped over the lazy dog
I come from a land down under
SambaNova makes extremely good software and hardware that's fun to use
```

will effectively be turned into this:

```
{"prompt": "", "completion": "The quick brown fox jumped over the lazy dog"}
{"prompt": "", "completion": "I come from a land down under"}
{"prompt": "", "completion": "SambaNova makes extremely good software and hardware that's fun to use"}
```

When processing text files, each line should be considered a *"data point"*. Depending on the `input_packing_config` parameter, these *"data points"* will be processed (and possibly combined) into sequences that are put in the **completion**. There is more information on the `input_packing_config` [below](#input_packing_config).

</br>

## Output
The `output_path` should be a directory that will contain all the tokenized HDF5 split files, and a sub-directory called `tokenizer`. This output directory constitutes a processed dataset and can be used for training a model after uploading to SambaStudio. The `tokenizer` sub-directory will be transferred to any output checkpoints that are saved by Sambastudio for the tokenizer to be used for inference later on.

### Holdout Evaluation Data
To evaluate on a holdout set of data during training, `pipeline.py` can create splits of holdout evaluation data.

To do this, include flags from only one of the two options below, only use one option or the other. Please review the [Flags](#flags) section for in detail descriptions of these flags.
- To specify the number of training splits and evaluation splits directly, use the three flags `--num_training_splits=...`, `--num_dev_splits=...` and `--num_test_splits=...`
- To specify the percentage of the data heldout for evaluation, you can specify `--dev_ratio=0.1` and `--test_ratio=...`, where 0.1 means that approximately 10% of the data will be included in the evaluation splits. You can also specify the `--num_training_splits=...` flag to control the total number of training splits, but we recommend to let this default.

All this evaluation data will saved under the `output_path`, if you want to run evaluation on the eval_splits during training you must enable `do_eval` on SambaStudio.

### Holdout Test Data
To create a holdout set of test data that is not tokenized, pipeline.py can create these splits and will leave the data un-tokenized and save it in the `output_path/test` directory. This data is left in jsonl text format because running evaluation or inference usually requires text inputs instead of tokenized inputs.

To do this, include flags from only one of the two options below, only use one option or the other. Please review the [Flags](#tokenize-one-file-flags) section for in detail descriptions of these flags.
- To specify the number of training splits and test splits directly, use the three flags `--num_training_splits=...`, `--num_dev_splits=...` and `--num_test_splits=...`
- To specify the percentage of the data heldout for testing, you can specify `--dev_ratio=...` and `--test_ratio=0.1`, where 0.1 means that approximately 10% of the data will be included in the test splits. You can also specify the `--num_training_splits=...` flag to control the total number of training splits, but we recommend to let this default.

### View Decoded HDF5 Files

If you want to view the contents of a processed dataset, you can decode an HDF5 file into a human readable text format. To do so, run the following command:

```python
python3 generative_data_prep/utils/decode_hdf5.py --hdf5_file_path=<PATH TO HDF5 FILE> --output_decoded_file_path=<PATH TO OUTPUT TXT FILE>
```

### Dataset Size Requirements
<details>

1. You need to ensure your dataset is large enough to run one batch of training.
2. Make sure that the number of sequences in the output dataset files satisfy this by checking `max_batch_size_train` in the `<OUTPUT_DIR>/metadata.yaml` file.
3. Use this value to set `batch_size` accordingly when starting a training job!

#### How to Check and Set

When starting a training job, ensure that the `batch_size` hyper-parameter is __*no bigger*__ than the `max_batch_size_train` shown in `metadata.yaml`.

For example:
```(shell)
$ cat <PROCESSED DATA DIRECTORY>/metadata.yaml

max_batch_size_dev: null
max_batch_size_train: 7
max_seq_length: 1024
number_of_dev_files: 0
number_of_test_files: 0
number_of_training_files: 32
token_type_ids: true
tokenizer_model_type: <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'>
vocab_size: 50257
```
Here you can see that `max_batch_size_train` is 7, so the `batch size` hyper-parameter cannot be greater than 7.

#### Explanation

With a sufficiently large dataset, you are generally fine with the defaults and can ignore. However, when the provided dataset is small (~1000 data points or less), you need to set the above values correctly or else you will likely run into a training error.

<br /> The dataset that you are providing will be split up across multiple hdf5 files based on the input parameters of the `pipeline` command.

* `max_seq_length` - The maximum sequence length the model you are using can take for a single data point. See more in [flags](#flags) section.
* `input_packing_config` - Determines how to pack the provided data into sequences that will be split across the hdf5 files for training. See more in the [flags](#flags) section.

Based on the size and strucutre of the dataset provided + these parameter settings, a different `max_batch_size_train` will be shown in `metadata.yaml` which dictates how large you can set the corresponding `batch_size` hyper-parameter setting when starting a model training job!

**_Note:_**: Not all models trained in studio will expose the `batch_size` parameter. For those that don't you should ensure your `max_batch_size_train` is larger than the default batch size (generally 16).
</details>

### Additional Details
<details>

If you include the `keep_split_jsonls` flag, then the `output_path` will additionally contain a `splits` directory that saves the jsonl versions of the HDF5 files, meaning that splits/train_1_of_X.jsonl is the jsonl text version of train_1_of_X.hdf5.

The output HDF5 files each contain two datasets:
- *input_ids*: sequences of tokens ids
- *token_type_ids*: describe the type of each token. The default id assignments are:
  - id=0 for tokens in the prompt
  - id=1 for tokens in the completion
  - id=2 for \<eos\> tokens that serve as padding tokens (will not be trained to predict)
  - id=3 for \<eos\> tokens at the end of articles, that define the attention boundary when training with article attention

</details>

</br>

## Flags

This section outlines all the flags you can set to customize the data prep pipeline for your use case!

| Flag Name  | Type | Default | Options | Description |
| --- | --- | --- | --- | --- |
| `input_file_path`  | str | REQUIRED | Any existing file path | Path to the input dataset which must be in `.jsonl` or `.txt` format. If dataset is in `.jsonl` format, the dataset needs to conform to the structure specified in the [Input](#input) section.|
| `output_path` | str | `input_file_path`'s directory | Any valid directory path | The directory to store the output files |
| `log_file_path` | str | `output_path`/logs.log | Any valid file path | The file to save the logs in, this will save the date and time, git commit hash, input arguments and metrics associated with the dataset. |
| `overwrite_output_path` | bool | False | Include flag for True, no arguments | Permission to delete and overwrite files in `output_path`. |
| `pretrained_tokenizer` | str | None | Valid tokenizer key from Huggingface | The pretrained tokenizer to be used, loaded using transformers.AutoTokenizer.from_pretrained(args.pretrained_tokenizer), in lieu of a `tokenizer_class`, `vocab_file` and `merges_file`. |
| `tokenizer_class` | str | 'gpt2' | ['gpt2'] | Tokenizer class to use, defaults to "gpt2" (transformers.GPT2Tokenizer). If `pretrained_tokenizer` is not specified, this is required. |
| `vocab_file` | str | None | Valid file path | The vocabulary file for the tokenizer. Should be a .json file for the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` is not specified, this is required. It should be a .json file for a GPT2 tokenizer. |
| `merges_file` | str | None | Valid file path | The merges file to be used with the tokenizer class specified by `tokenizer_class`. If `pretrained_tokenizer` tokenizer is not specified, this is required. It should be a .txt file for a GPT2 tokenizer. |
| `special_tokens_dict` | str | None | string representation of json | Any non-standard special tokens in JSON format to add to tokenizer. e.g. \"{'sep_token': \"[SEP]\"}\". Additional tokens can be also added using the "additional_special_tokens" keyword. For example, indentation encoding can be added with \"{'additional_special_tokens': [\"\t\", \"\t\t\", \"\t\t\t\"]}\". |
| `max_seq_length` | int | 2048 | 512 for gpt2 small, 1024 for gpt-xl, 2048 for gpt3-13B. | The maximum sequence length of the model you are using. |
| `input_packing_config` <span id="input_packing_config"></span> | str | 'full' | ['full', 'single::truncate_left', 'single::truncate_right', 'single::drop', 'greedy::truncate_left', 'greedy::truncate_right', 'greedy::drop'] | The first argument in the packing config defines the method of placing text into sequences, the second argument defines how to handle jsonls that do not fit within the max_seq_length. `'full'`: Defines the entire packing config, Completely fill sequences with tokens, as soon as sequences is full start packing into new sequence. Ignore article boundaries, they may be split across multiple sequences. `'greedy'`: Fit as many articles as possible into a sequence, make sure no article is split across multiple sequences. Fill the left over space in each sequence with padding. `'single'`: Each sequence contains only 1 article.  Fill the rest of the sequence with padding.  `'drop'`: Drop the entire article if there are any tokens that overflow beyond the max sequence length.  `'truncate_left'`:  Truncate the article from the left if there are any tokens that overflow beyond the max sequence length.  `'truncate_right'`:  Truncate the article from the right if there are any tokens that overflow beyond the max sequence length. |
| `packing_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | 'jsonl': When packing text into sequences, keeps json lines together. This means that for greedy or single packing if the entire line does not fit in the sequences it will be thrown out. 'prompt_completion_pair': When packing text into sequences, prompt_completion_pairs together, but may break up json lines that contain a list of prompt completion pairs. |
| `attention_boundary` | str | 'jsonl' | ['jsonl', 'prompt_completion_pair'] | The boundary to use when training with --article_attention flag. If you choose prompt_completion_pair tokens will only attend to tokens in the prompt_completion_pair. If you choose jsonl, then tokens will attend to all the prompt completion pairs in the jsonl |
| `prompt_keyword` <span id="prompt_keyword"></span> | str | 'prompt' | | If your input json has a string keyword for prompt other than "prompt", place the keyword here. e.g Input_json: {"source": ... "target": ...} ->`prompt_keyword`='source'. |
| `completion_keyword` | str | 'completion' | | If your input json has a string keyword for completion other than "completion", place the  keyword here. e.g Input_json: {"source": ... "target": ...} -> --completion_keyword='target'. |
| `prompt_prefix` | str | 'None' | | text to add before the prompt, for chatML conventions use (e.g. "\<human\>:") |
| `prompt_postfix` | str | 'None' | | text to add after the prompt, for chatML conventions use (e.g. "\<bot\>:") |
| `disable_space_separator` | bool | False | Include flag for True, no arguments |  If you include this flag, NO spaces will be prepended to the completion. (If you do not add this flag then a space is added to every completion if it does not already have a space). Including this flag is dangerous and not recommended because if you have input data like {"prompt": "hello." "completion": "how are you?"}, when the prompt and completion are combined it will look like "hello.how are you?" which will mess up the tokenization.--completion_keyword='target'. |
| `keep_prompt_only_sequences` | bool | False | Include flag for True, no arguments | If you include this flag, packed sequences with only prompt tokens will not be dropped. Data with only prompt will be dropped by default because training with prompt-only sequences with prompt_loss_weight=0.0 may lead to errors. Data is dropped because of one of the following conditions: 1. the input file data prompt completion pairs contains only a prompt. 2. If the sequence is truncated such that only prompt tokens remain |
| `categories_path` | str | False | If you include this flag, then the 'category' field from your input jsonls will be stored in the 'category_id' dataset in your output hdf5 files. This flag must point to the file path of a json file that contains a list of all the strings of the 'category' keys in your dataset.|
| `shuffle` <span id="shuffle"></span> | str | 'False' | ['False', 'on_RAM', 'large_file'] | Choose the on_RAM option if your file is small enough to fit on RAM (If you are not sure if it fits on RAM, you can probably use this flag). If you are running a linux operating system and your file is too large to fit on RAM, please choose large_file option, this will run approximate file shuffling that can handle files of any size. If you want to do large file shuffling but you are not on linux, please shuffle the file before using this script. If the input file should not be shuffled, do not include this flag, it defaults to False. |
| `num_training_splits` | int | 32 if input_file_size < 10GB, 128 if 10GB < input_file_size <100GB, 256 if 100GB < input_file_size | | The number of training files to split input data into. We recommend you do not include this flag and allow it to default. If you do not default this flag, you have two options. Option 1: specify this flag with the `dev_ratio` and `test_ratio` flags, The total number of splits will be (`num_training_splits` / (1-`dev_ratio`-`test_ratio`)), and the number of dev and test splits are calculated accordingly. Option 2: specify this flag with the `num_dev_splits` and `num_test_splits` flags which define the number of splits directly. NOTE: the number of training splits must be greater than the number of training workers you have, and we recommend that the number of splits is a multiple of the number of workers you have. |
| `dev_ratio` | float | 0.0 | [0 - 1] | The ratio of data that should be excluded from train set and used for evaluation, defaults to 0%. If you specify this flag, do not specify `num_dev_splits` or `num_test_splits`. |
| `test_ratio` | float | 0.0 | [0 - 1] | The ratio of data that should be excluded from train set and is saved for testing. This data is not tokenized and left in jsonline format, defaults to 0%. If you specify this flag, do not specify `num_dev_splits` or `num_test_splits`. |
| `num_dev_splits` | int | None | Any int | number of dev (eval) splits. If you do not specify `dev_ratio`, you may specify this flag. If you include this flag, you must also include the `num_test_splits` and `num_training_splits` flags. |
| `num_test_splits` | int | None | Any int | Number of test splits. If you do not specify `test_ratio`, you may specify num_test_splits. If you include this flag, you must also include the `num_dev_splits` and `num_training_splits` flags. |
| `do_not_balance_hdf5` | bool | False | Include flag for True, no arguments | Include this flag if you DO NOT want to balance HDF5 files, this is not recommended unless the you are dealing with a huge amount of data (many terra bytes), or do not want shuffling between splits. |
| `keep_split_jsonls` | bool | False | Include flag for True, no arguments | If you DO NOT want to delete split jsonls files that are in text format in the `output_path/splits` directory include this flag. The only reason you would include this flag is if you want to see what text is in each HDF5, meaning that splits/train_1_of_X.jsonl is the jsonl text version of train_1_of_X.hdf5. Including this flag will increase the storage space of your dataset by more than two times. |
| `num_workers` | int | False | 0 <= `num_workers`<= # of available CPUs | The number of CPU workers to run tokenization with, if the previous run failed due to OOM, you need to decrease this number. |

</br>

## Examples


### Fine-tuning
Fine-tuning (also known as "generative tuning") is a technique used to adapt a pre-trained language model to perform better at a specific task. This approach typically involves training the model on input data that is structured as a "prompt" followed by a "completion". The prompt represents the input for a specific task, while the completion is the output that the model should generate. During training, the model learns to generate the relevant completion tokens based on the context provided by the prompt tokens.

The benefit of using this training format is that the model can learn to generate high-quality outputs for a specific task without requiring a large amount of task-specific training data. By leveraging the pre-trained language model's knowledge gained from being trained on a large corpus of text data, the fine-tuned model can quickly adapt to the new task and generate high-quality outputs with minimal training data.

When training on this kind of data using SambaStudio, set `prompt_loss_weight=0.0`. This ensures that the model does not learn to generate the prompt tokens, and only learns to generated completion tokens.

#### Example data
For fine-tuning, your data should be in `.jsonl` format with prompts and completions designed for the task you're adapting to.
> [Jsonlines with a prompt and completion](tests/examples/generative_tuning/example_generative_tuning_data.jsonl)

#### Example command

```python
python3 -m generative_data_prep pipeline --input_file_path=./tests/examples/generative_tuning/example_generative_tuning_data.jsonl --output_path=./tests/examples/generative_tuning/pipelined_generative_tuning --pretrained_tokenizer=gpt2 --max_seq_length=1024 --shuffle=on_RAM --input_packing_config=single::drop
```

> [View decoded output](tests/examples/generative_tuning/decoded_data_prepped_generative_tuning.txt)


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

</br>

## Understanding Command Outputs

### Terminal Output
The metrics associated with this dataset will be printed in the terminal as well as being logged at `<OUTPUT DIR PATH>/logs.log`. These metrics give some insight into how the data was packed into sequences, and information about the training dataset.
| Metric Name      | Definition | How to Interpret? |
| --------- | --------- | --------- |
| Articles | The number of lines in the input dataset. | How many text documents in the input dataset. |
| Dataset Tokens | Number of tokens in the output hdf5 dataset. | How many tokens are in the training dataset. But this includes both prompt tokens and padding tokens, so this metric does not necessarily show how many tokens will learned by the model. |
| Prompt Tokens | Number of prompt tokens in the output hdf5 dataset. | <- |
| Completion Tokens | Number of completion tokens in the output hdf5 dataset. | <- |
| Padding Tokens | Number of padding tokens in the output hdf5 dataset. | <- |
| Average Completion Length | Number of completion tokens divided by number of input articles. | The length of the average completion in the dataset. |
| Average Prompt Length | Number of prompt tokens divided by number of input articles. |  The length of the average prompt in the dataset. |
| Data Utilization | Percent of non-padding tokens in output HDF5 dataset divided by number of tokens in input dataset. | This metric reveals how much of the input data makes it to the output dataset. If this percent is much less than 100%, that means a lot of the input data will not be trained on. Refer to the "Dropped From Packing" or "Dropped From All Prompt" metrics to see why this is happening. |
| Dropped From Packing  | Number of tokens dropped during packing, divided by number of tokens in input dataset. | The percent of tokens are dropped because they do not fit into the sequence length, and the `input_packing_config` does not allow them to be overflowed.|
| Dropped From All Prompt | Number of tokens dropped because all the tokens in a sequence are prompt tokens, divided by the number of tokens in input dataset. | Sequences that are all prompts or padding (no completion tokens) are dropped. This is because the model will not learn anything from these sequences and the loss will be 0, which may cause errors. |
| Sequence Utilization | Average number of non-padding tokens in a sequence divided by sequence length. | The percent of the tokens in each sequence are actually used for training. This number imrpoved be changed by using different `input_packing_config`. The packing styles from highest sequence utilization to lowest are: `full`, `greedy::truncate_left` (or truncate_right), `greedy::drop`, `single::truncate_left` (or truncate_right), `single::drop`.|
| Seq Completion Utilization | Average number of completions tokens in a sequence divided by sequence length. | The percent of the tokens in a sequence are learned.|

### Metadata Output File

To help improve speed and cross-checking we provide a metadata file along with the dataset. This file is located right under the `output_dir` as `metadata.yaml`. This file is used along with a custom pydantic model which you can import from this library which will verify the dataset parameters and the training parameters. This can be used as a way to catch bugs before training begins.

```
max_seq_length: int
token_type_ids: bool
vocab_size: int
tokenizer_model_type: str
number_of_training_files: int
number_of_dev_files: int
number_of_test_files: int
max_batch_size_train: int
max_batch_size_dev: Optional[int]
```

NOTE:
* `tokenizer_model_type` is the string conversion of `type(modelConfig)`. Can use this field to compare the model used during training, which can be extracted by using `AutoConfig` in Huggingface transformers. Then wrapping it with `str(type())`.
* `max_batch_size_dev` will be `None` unless dev files are created during generative data pipeline.
* `token_type_ids` will always be `True` for now since they are always generated.


## Advanced Usage

The following are some advanced usage patterns that may be applicable to you. Follow the links for more information:

- If you have data that has been custom pre-split, and you would like to tokenize these files individually, check out the [Single File Tokenization Guide](./advanced_usage.md#tokenize-one-file-flags)
- If you want to build in custom dataset validation with our pydantic model, look at our section on [pydantic dataset validation](./advanced_usage.md#how-to-use-pydantic-model).
- If you want to build in dataset verification checks, look at our section on [checking for dataset corruption](./advanced_usage.md#how-to-check-for-corruption).
- If you want to contribute to this project, check out the [contribution section](./advanced_usage.md#contributing).
