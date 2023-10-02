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


This module implements ArticleTokenizer.

Article tokenizer takes any input text or jsonl, runs tokenization using
the passed in tokenizer, and then packs the tokens into sequences using
the SequencePacker class.
"""

from typing import TypeVar

# from tabulate import tabulate

# custom type representing subclasses of Metrics
MetricsSubClass = TypeVar("MetricsSubClass", bound="Metrics")


class Metrics:
    """Store all the metrics associated with a tokenization process."""

    def __init__(self):
        """Create a metrics tracking object."""
        self.tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.padding_tokens: int = 0
        self.sequences: int = 0
        self.articles: int = 0
        self.prompt_completion_pairs: int = 0
        self.prompt_completion_pairs: int = 0
        self.tokens_dropped: int = 0
        self.articles_dropped_from_packing: int = 0
        self.articles_dropped_from_all_prompt: int = 0

    def __iadd__(self: MetricsSubClass, new_metrics: "Metrics") -> MetricsSubClass:
        """Implement += for Metrics."""
        self.tokens += new_metrics.tokens
        self.prompt_tokens += new_metrics.prompt_tokens
        self.completion_tokens += new_metrics.completion_tokens
        self.padding_tokens += new_metrics.padding_tokens
        self.sequences += new_metrics.sequences
        self.articles += new_metrics.articles
        self.prompt_completion_pairs += new_metrics.prompt_completion_pairs
        self.tokens_dropped += new_metrics.tokens_dropped
        self.articles_dropped_from_packing += new_metrics.articles_dropped_from_packing
        self.articles_dropped_from_all_prompt += new_metrics.articles_dropped_from_all_prompt

        return self

    @property
    def percent_articles_dropped(self) -> float:
        """Percent of the articles dropped due to either packing or all prompt."""
        total_articles_dropped = self.articles_dropped_from_all_prompt + self.articles_dropped_from_packing
        return total_articles_dropped / self.articles

    @property
    def percent_articles_dropped_from_prompt(self) -> float:
        """The percent of the articles dropped due to having only prompt tokens (no completion)."""
        return self.articles_dropped_from_all_prompt / self.articles

    @property
    def percent_articles_dropped_from_packing(self) -> float:
        """The percent of the articles that are dropped due to packing style."""
        return self.articles_dropped_from_all_prompt / self.articles

    @property
    def averge_prompt_length(self) -> float:
        """The average number of tokens per prompt."""
        return self.prompt_tokens / self.articles

    @property
    def average_completion_length(self) -> float:
        """The average number of tokens per completion."""
        return self.completion_tokens / self.articles

    @property
    def sequence_utilization(self) -> float:
        """What percent of the tokens are not padding."""
        return 1 - (self.padding_tokens / self.tokens)

    @property
    def sequence_completion_utilization(self) -> float:
        """What percent of the tokens are completions."""
        return self.completion_tokens / self.tokens

    def _to_str_percent(self, value: int) -> str:
        percent = round(value * 100, 2)
        return f"{percent:.2f}%"

    def __str__(self):
        """String representation of metrics."""
        table = [
            ["Sequences", self.sequences],
            ["Articles", self.articles],
            ["Tokens", self.tokens],
            ["Prompt Tokens", self.prompt_tokens],
            ["Completion Tokens", self.completion_tokens],
            ["Padding Tokens", self.padding_tokens],
            ["Prompt Completion Pairs", self.prompt_completion_pairs],
            ["Tokens Dropped", self.tokens_dropped],
            ["Articles Dropped From Packing", self.articles_dropped_from_packing],
            ["Percent Dropped From Packing", self._to_str_percent(self.percent_articles_dropped_from_packing)],
            ["Articles Dropped From All Prompt", self.articles_dropped_from_all_prompt],
            ["Percent Dropped From All Prompt", self._to_str_percent(self.percent_articles_dropped_from_prompt)],
            ["Sequence Utilization", self._to_str_percent(self.sequence_utilization)],
            ["Sequence Completion Utilization", self._to_str_percent(self.sequence_completion_utilization)],
            ["Average Completion Length", round(self.average_completion_length, 2)],
            ["Average Prompt Length", round(self.averge_prompt_length, 2)],
        ]
        ret = "=================================================="
        for name, value in table:
            ret += "\n" + f"{name}: {value}"
        ret += "=================================================="
        return ret

        # return tabulate(table, "fancy_outline")
