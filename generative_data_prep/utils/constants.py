"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from enum import Enum


class BaseEnum(Enum):
    """ Contains additional utility methods for the custom Enums """
    @classmethod
    def as_list(cls):
        return [member.value for member in cls]


class PackingStyleType(str, BaseEnum):
    FULL = 'full'
    SINGLE_TRUNCATE_OVERFLOW = 'single_truncate_overflow'
    SINGLE_DROP_OVERFLOW = 'single_drop_overflow'
    GREEDY = 'greedy'


class BoundaryType(str, BaseEnum):
    PROMPT_COMPLETION_PAIR = 'prompt_completion_pair'
    JSONL = 'jsonl'


class TokenTypeIds(int, BaseEnum):
    PROMPT = 0
    COMPLETION = 1
    PADDING = 2
    SEP = 3
