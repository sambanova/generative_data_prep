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


class BoundaryType(str, BaseEnum):
    PROMPT_COMPLETION_PAIR = 'prompt_completion_pair'
    JSONL = 'jsonl'


class OverflowType(str, BaseEnum):
    DROP = 'drop'
    TRUNCATE_LEFT = 'truncate_left'
    TRUNCATE_RIGHT = 'truncate_right'


class PackingStyleType(str, BaseEnum):
    GREEDY = 'greedy'
    FULL = 'full'
    SINGLE = 'single'


class TokenTypeIds(int, BaseEnum):
    PROMPT = 0
    COMPLETION = 1
    PADDING = 2
    SEP = 3
