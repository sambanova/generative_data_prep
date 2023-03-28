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


Abstract class for text buffers, which allow the reading and writing of text.
"""

from abc import ABC, abstractmethod
from typing import Any


class FileBuffer(ABC):
    """Represent a data structure that stores text between pipeline processing stages."""
    @abstractmethod
    def read(self) -> str:
        """Read a line from the TextBuffer."""
        raise NotImplementedError

    @abstractmethod
    def write(self, line: Any):
        """Write a line to the TextBuffer.

        Args:
            line: The line to be written to the TextBuffer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_concurrent(self) -> bool:
        """Return whether this text buffer supports concurrent reads and writes or not."""
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        """Called when you enter the text buffer using "with TextBuffer() as f:" """
        raise NotImplementedError

    @abstractmethod
    def __exit__(self):
        """Called when you exit the text buffer by exiting the with block"""
        raise NotImplementedError
