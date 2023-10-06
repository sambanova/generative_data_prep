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


This module implements Metrics.

A metrics object keeps track of all the information about a tokenized dataset when it is created.

"""

from typing import TypeVar

from tabulate import tabulate

# custom type representing subclasses of Metrics
MetricsSubClass = TypeVar("MetricsSubClass", bound="Metrics")


class Metrics:
    """Store all the metrics associated with a tokenization process."""

    def __init__(self):
        """Create a metrics tracking object."""
        self.input_tokens: int = 0  # how many tokens are in the input jsonl dataset
        self.output_tokens: int = 0  # how many tokens are in the output hdf5 dataset
        self.prompt_tokens: int = 0  # how many prompt tokens are in the output hdf5 dataset
        self.completion_tokens: int = 0  # how many completion tokens are in the output hdf5 dataset
        self.padding_tokens: int = 0  # how many padding tokens are in the output hdf5 dataset
        self.sequences: int = 0  # how many sequences are in the output hdf5 dataset
        self.articles: int = 0  # how many articles are in the input dataset
        self.tokens_dropped_from_packing: int = 0  # how many tokens are dropped because of packing
        self.tokens_dropped_from_all_prompt: int = (
            0  # how many tokens are dropped because no completions in the entire sequence
        )

    def __iadd__(self: MetricsSubClass, new_metrics: "Metrics") -> MetricsSubClass:
        """Implement += for Metrics."""
        self.input_tokens += new_metrics.input_tokens
        self.output_tokens += new_metrics.output_tokens
        self.prompt_tokens += new_metrics.prompt_tokens
        self.completion_tokens += new_metrics.completion_tokens
        self.padding_tokens += new_metrics.padding_tokens
        self.sequences += new_metrics.sequences
        self.articles += new_metrics.articles
        self.tokens_dropped_from_packing += new_metrics.tokens_dropped_from_packing
        self.tokens_dropped_from_all_prompt += new_metrics.tokens_dropped_from_all_prompt

        return self

    @property
    def percent_tokens_dropped_from_all_prompt(self) -> float:
        """The percent of the articles dropped due to having only prompt tokens (no completion)."""
        return self.tokens_dropped_from_all_prompt / self.input_tokens

    @property
    def percent_tokens_dropped_from_packing(self) -> float:
        """The percent of the articles that are dropped due to packing style."""
        return self.tokens_dropped_from_packing / self.input_tokens

    @property
    def averge_prompt_length(self) -> float:
        """The average number of tokens per prompt."""
        return self.prompt_tokens / self.articles

    @property
    def average_completion_length(self) -> float:
        """The average number of tokens per completion."""
        return self.completion_tokens / self.articles

    @property
    def data_utilization(self) -> float:
        """What percent of the input dataset tokens are actually included."""
        return (self.prompt_tokens + self.completion_tokens) / self.input_tokens

    @property
    def sequence_utilization(self) -> float:
        """What percent of the tokens are not padding ie what percent of tokens are prompt or completion."""
        return (self.prompt_tokens + self.completion_tokens) / self.output_tokens

    @property
    def sequence_completion_utilization(self) -> float:
        """What percent of the tokens are completions."""
        return self.completion_tokens / self.output_tokens

    def _to_str_percent(self, value: int) -> str:
        percent = round(value * 100, 2)
        return f"{percent:.2f}%"

    def __str__(self):
        """String representation of metrics."""
        table = [
            ["Sequences", self.sequences],
            ["Articles", self.articles],
            ["Dataset Tokens", self.output_tokens],
            ["Prompt Tokens", self.prompt_tokens],
            ["Completion Tokens", self.completion_tokens],
            ["Padding Tokens", self.padding_tokens],
            ["Average Completion Length", round(self.average_completion_length, 2)],
            ["Average Prompt Length", round(self.averge_prompt_length, 2)],
            ["Data Utilization", self._to_str_percent(self.data_utilization)],
            ["Dropped From Packing", self._to_str_percent(self.percent_tokens_dropped_from_packing)],
            ["Dropped From All Prompt", self._to_str_percent(self.percent_tokens_dropped_from_all_prompt)],
            ["Sequence Utilization", self._to_str_percent(self.sequence_utilization)],
            ["Seq Completion Utilization", self._to_str_percent(self.sequence_completion_utilization)],
        ]

        return tabulate(table, tablefmt="fancy_grid")
