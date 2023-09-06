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


Implement a Text Buffer for writing tokenized sequences to hdf5 files.
"""

from types import TracebackType
from typing import List, Optional, Type

import h5py
import numpy as np

from generative_data_prep.tokenized_line import TokenizedSequence

from .file_buffer import FileBuffer

MEGABYTE = 1048576


class Hdf5FileBuffer(FileBuffer):
    """Implementation of Hdf5TextBuffer to write tokenized sequences to hdf5 files."""

    def __init__(
        self,
        hdf5_file_path: str,
        max_seq_length: int,
        dump_categories: Optional[bool] = False,
        data_type: Optional[str] = "i4",
        max_chunk_size: Optional[int] = MEGABYTE,
    ):
        """Initialize Hdf5TextBuffer.

        Args:
            hdf5_file_path: Path to the hdf5 file to write to
            max_seq_length: Maximum sequence length of sequences that will be written
            dump_categories: Whether to dump category_ids into hdf5 files.
            data_type: Data type to write into hdf5 file. Defaults to 'i4'.
            max_chunk_size: How large chunks to write into hdf5 at once. Defaults to 1 MEGABYTE.
        """
        self.hdf5_file_path = hdf5_file_path
        self.max_seq_length = max_seq_length
        self.dump_categories = dump_categories
        self.data_type = data_type
        data_type_size = np.dtype(data_type).itemsize
        self.max_chunk_length = int(max_chunk_size / (max_seq_length * data_type_size))
        self._chunk: List[TokenizedSequence] = []
        self.first_dump = True

    def __enter__(self):
        """Open hdf5 file when Hdf5TextBuffer is accessed with Hdf5TextBuffer() as ...

        Returns:
            self, this object
        """
        self.hdf5_file = h5py.File(self.hdf5_file_path, "w")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """When the with Hdf5TextBuffer() as ... is exited, flush and close hdf5 file.

        Args:
            exc_type: exception type
            exc_val: exception value
            exc_tb: exception throwback

        Raises:
            RuntimeError: If there is an exception
        """
        if exc_type is not None or exc_val is not None or exc_tb is not None:
            err_msg = f"Hdf5TextBuffer exit failed with exc_type: {exc_type}, exc_val: {exc_val}, exc_tb: {exc_tb}"
            raise RuntimeError(err_msg)

        if len(self._chunk) > 0:
            self._dump_chunk(self._chunk)
        self._chunk = []

        self.hdf5_file.flush()
        self.hdf5_file.close()

        return True

    def _dump_data(self, dataset_name, new_shape, data):
        """Resizes self.hdf5_file to new_shape then dumps data into it.

        Args:
            dataset_name: string to index index into hdf5 dataset
            new_shape: New shape to resize dataset
            data: Data to add into self.hdf5_file['dataset_name']
        """
        self.hdf5_file[dataset_name].resize(new_shape)
        self.hdf5_file[dataset_name][-len(data) :] = data

    def _dump_chunk(self, chunk):
        """Add the data from chunk into self.hdf5_file.

        Args:
            chunk: A chunk of tokenized sequences to add to self.hdf5_file
        """
        if len(chunk) > self.max_chunk_length:
            err_msg = f"Trying to dump a chunk of size {len(chunk)} into Hdf5TextBuffer, "
            err_msg += f"which is larger than max_chunk_size {self.max_chunk_length}"
            raise ValueError(err_msg)

        for tokenized_seq in chunk:
            if len(tokenized_seq) != self.max_seq_length:
                raise ValueError("tokenized sequence has not been filled")

        dump_token_ids = list(map(lambda seq: seq.dump_token_ids(), chunk))
        dump_token_type_ids = list(map(lambda seq: seq.dump_token_type_ids(), chunk))
        if self.dump_categories:
            dump_category_ids = list(map(lambda seq: seq.dump_category_ids(), chunk))

        if self.first_dump:
            self.hdf5_file.create_dataset(
                "input_ids",
                data=dump_token_ids,
                dtype=self.data_type,
                compression="gzip",
                maxshape=(None, self.max_seq_length),
            )
            self.hdf5_file.create_dataset(
                "token_type_ids",
                data=dump_token_type_ids,
                dtype=self.data_type,
                compression="gzip",
                maxshape=(None, self.max_seq_length),
            )
            if self.dump_categories:
                self.hdf5_file.create_dataset(
                    "category_ids",
                    data=dump_category_ids,
                    dtype=self.data_type,
                    compression="gzip",
                    maxshape=(None, self.max_seq_length),
                )
        else:
            num_dump_seq = len(chunk)
            new_shape = (
                self.hdf5_file["input_ids"].shape[0] + num_dump_seq,
                self.max_seq_length,
            )

            self._dump_data("input_ids", new_shape, dump_token_ids)
            self._dump_data("token_type_ids", new_shape, dump_token_type_ids)
            if self.dump_categories:
                self._dump_data("category_ids", new_shape, dump_category_ids)

        self.first_dump = False

    def read(self) -> str:
        """Not implemented."""
        raise NotImplementedError

    def write(self, tokenized_sequences: List[TokenizedSequence]):
        """Save tokenized_sequences to self._chunk, if self._chunk is full then call self._dump_chunk.

        Args:
            tokenized_sequences: Tokenized sequences to save and when chunk is full write to hdf5.
        """
        self._chunk += tokenized_sequences
        while len(self._chunk) >= self.max_chunk_length:
            self._dump_chunk(self._chunk[: self.max_chunk_length])
            self._chunk = self._chunk[self.max_chunk_length :]

    @property
    def is_concurrent(self) -> bool:
        """Returns true if this TextBuffer is concurrent.

        Returns:
            False, since Hdf5TextBuffer is not concurrent
        """
        return False
