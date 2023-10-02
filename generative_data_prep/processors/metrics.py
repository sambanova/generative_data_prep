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

from generative_data_prep.utils.utils import SEP_STR

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
        return round(total_articles_dropped / self.articles, 2)

    @property
    def percent_articles_dropped_from_prompt(self) -> float:
        """The percent of the articles dropped due to having only prompt tokens (no completion)."""
        return round(self.articles_dropped_from_all_prompt / self.articles, 2)

    @property
    def percent_articles_dropped_from_packing(self) -> float:
        """The percent of the articles that are dropped due to packing style."""
        return round(self.articles_dropped_from_all_prompt / self.articles, 2)

    @property
    def averge_prompt_length(self) -> float:
        """The average number of tokens per prompt."""
        return round(self.prompt_tokens / self.articles, 2)

    @property
    def average_completion_length(self) -> float:
        """The average number of tokens per completion."""
        return round(self.completion_tokens / self.articles, 2)

    @property
    def sequence_utilization(self) -> float:
        """What percent of the tokens are not padding."""
        return round(1 - (self.padding_tokens / self.tokens), 2)

    @property
    def sequence_completion_utilization(self) -> float:
        """What percent of the tokens are completions."""
        return round(self.completion_tokens / self.tokens, 2)

    def __str__(self):
        """String representation of metrics."""
        ret = SEP_STR
        ret += "\n" + f"Sequences: {self.sequences}"
        ret += "\n" + f"Articles: {self.articles}"
        ret += "\n" + f"Tokens: {self.tokens}"
        ret += "\n" + f"Prompt Tokens: {self.prompt_tokens}"
        ret += "\n" + f"Completion Tokens: {self.completion_tokens}"
        ret += "\n" + f"Padding Tokens: {self.padding_tokens}"
        ret += "\n" + f"Prompt Completion Pairs: {self.prompt_completion_pairs}"
        ret += "\n" + f"Tokens Dropped: {self.tokens_dropped}"
        ret += "\n" + f"Articles Dropped From Packing: {self.articles_dropped_from_packing}"
        ret += "\n" + f"Percent Dropped From Packing: {self.percent_articles_dropped_from_packing}"
        ret += "\n" + f"Articles Dropped From All Prompt: {self.articles_dropped_from_all_prompt}"
        ret += "\n" + f"Percent Dropped From All Prompt: {self.percent_articles_dropped_from_prompt}"
        ret += "\n" + f"Sequence Utilization: {self.sequence_utilization}"
        ret += "\n" + f"Sequence Completion Utilization: {self.sequence_completion_utilization}"
        ret += "\n" + f"Average Completion Length: {self.average_completion_length}"
        ret += "\n" + f"Average Prompt Length: {self.averge_prompt_length}"

        return ret
