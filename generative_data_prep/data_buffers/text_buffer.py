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

from data_buffer import DataBuffer
from typing import Any, Optional
from utils import ShuffleType


class TextBuffer(DataBuffer):
    """Represent a data structure that stores text between pipeline processing stages."""
    def __init__(self, num_readers: Optional[int] = None, num_writers: Optional[int]=None, input_path: Optional[str]=None, output_path: Optional[str] = None, shuffle_type: Optional[ShuffleType]=None):
        self.num_reader = num_readers
        self.num_writers = num_writers
        self.input_path = input_path
        self.output_path = output_path
        self.shuffle_type = shuffle_type

        # If shuffle is not None, shuffle in the correct fashion
        # assert input path is specified
        

        # Split into num_reader and store into output path? 
        # If shuffled was called, delete the temp shuffled file

        # store a list of files
    

    def __enter__(self):
        """Open hdf5 file when Hdf5TextBuffer is accessed with Hdf5TextBuffer() as ...

        Returns:
            self, this object
        """
        # open all the split files and store a list of open files that you can read from by index


    def __exit__(self, exc_type, exc_val, exc_tb):
        """When the with Hdf5TextBuffer() as ... is exited, flush and close hdf5 file.

        Args:
            exc_type: exception type
            exc_val: exception value
            exc_tb: exception throwback

        Raises:
            RuntimeError: If there is an exception
        """
        # iterate through the open files and flush and close each one.
        pass

    def read(self) -> str:
        """Read a line from the TextBuffer."""
        raise NotImplementedError

    def write(self, line: Any):
        """Write a line to the TextBuffer.

        Args:
            line: The line to be written to the TextBuffer.
        """
        raise NotImplementedError

    def is_concurrent(self) -> bool:
        """Return whether this text buffer supports concurrent reads and writes or not."""
        return True
