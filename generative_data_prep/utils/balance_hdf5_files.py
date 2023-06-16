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

Function to balance the number of sequences in hdf5 splits.
"""

import sys
import time
from glob import glob
from typing import List

import h5py
import numpy as np


def balance_hdf5_files(hdf5_file_paths: List[str]) -> None:
    """Balances all the files in hdf5_file_paths, to have the same number fo sequences (within 1).

    This is done by first calculating the average number of sequences per file,
    and the remainder of sequences. Then all the hdf5 files are iterated over,
    and if they have more than the average number of sequences, those sequences are
    removed from the hdf5 file and saved. If the hdf5 file has fewer sequences than
    the average, then we save how many more sequences the file needs. The last step
    is to iterate over all the files that need sequences, and add the sequences
    from the saved extra sequences of the other hdf5 files.

    Instead of dropping the remainder sequences, the first [remainder] files iterated
    over are granted 1 extra sequence. This means that the files are not perfectly
    balanced, but are within 1 sequence of each other.

    Args:
        hdf5_file_paths: A list of the hdf5 files to balance
    """
    num_files = len(hdf5_file_paths)
    if num_files == 0:
        return

    # Iterate over all the splits to calculate average number of sequences there should be per file
    tot_seqs = 0
    for file_path in hdf5_file_paths:
        with h5py.File(file_path) as curr_hdf5_file:
            tot_seqs += curr_hdf5_file["input_ids"].shape[0]
    num_files = len(hdf5_file_paths)
    avg_seqs = int(tot_seqs / num_files)
    remainder = tot_seqs % num_files
    assert (
        avg_seqs * num_files + remainder == tot_seqs
    ), f"avg_seqs * num_files + remainder = {avg_seqs * num_files + remainder} but tot_seqs = {tot_seqs}, not equal!"

    # Remove and save sequences from files with more than the average
    # Determine how many more sequences files with less than the average need
    file_path_to_num_needed_seqs = {}
    extra_token_seqs = []
    extra_ttid_seqs = []
    for file_path in hdf5_file_paths:
        with h5py.File(file_path, "r+") as curr_hdf5_file:
            curr_num_seq = curr_hdf5_file["input_ids"].shape[0]
            seq_len = curr_hdf5_file["input_ids"].shape[1]
            # If file has more than the average number of sequences
            if curr_num_seq > avg_seqs:
                num_extra_seq = curr_num_seq - avg_seqs
                if remainder > 0:
                    remainder -= 1
                    num_extra_seq -= 1
                    if num_extra_seq == 0:
                        continue
                curr_num_seq -= num_extra_seq
                # save extra token sequences, and update hdf5 file
                curr_extra_token_seqs = curr_hdf5_file["input_ids"][-num_extra_seq:]
                extra_token_seqs.append(curr_extra_token_seqs)
                curr_hdf5_file["input_ids"].resize((curr_num_seq, seq_len))

                # save extra token type id sequences, and update hdf5 file
                curr_extra_ttid_seqs = curr_hdf5_file["token_type_ids"][-num_extra_seq:]
                extra_ttid_seqs.append(curr_extra_ttid_seqs)
                curr_hdf5_file["token_type_ids"].resize((curr_num_seq, seq_len))

            # If file has less than the average number of sequences
            elif curr_num_seq < avg_seqs:
                num_needed_seq = avg_seqs - curr_num_seq
                if remainder > 0:
                    num_needed_seq += 1
                    remainder -= 1
                # store how many more sequences the file needs to be average
                file_path_to_num_needed_seqs[file_path] = num_needed_seq

            elif curr_num_seq == avg_seqs:
                if remainder > 0:
                    remainder -= 1
                    file_path_to_num_needed_seqs[file_path] = 1

    if len(extra_token_seqs) > 0:
        assert len(extra_token_seqs) == len(extra_ttid_seqs)
        extra_token_seqs_np = np.vstack(extra_token_seqs)
        extra_ttid_seqs_np = np.vstack(extra_ttid_seqs)
    else:
        extra_token_seqs_np = np.zeros((0, 0))
        extra_ttid_seqs_np = np.zeros((0, 0))
        assert (
            len(file_path_to_num_needed_seqs) == 0
        ), f"No extra sequences found, but these files need extra sequences {file_path_to_num_needed_seqs}"

    # Iterate through all the files that need more sequences
    for file_path, num_needed_seq in file_path_to_num_needed_seqs.items():
        with h5py.File(file_path, "r+") as curr_hdf5_file:
            # new shape for hdf5 file after sequences have been added
            new_shape = (
                curr_hdf5_file["input_ids"].shape[0] + num_needed_seq,
                curr_hdf5_file["input_ids"].shape[1],
            )

            # add extra token sequences to hdf5 file
            curr_hdf5_file["input_ids"].resize(new_shape)
            curr_hdf5_file["input_ids"][-num_needed_seq:] = extra_token_seqs_np[
                :num_needed_seq
            ]
            # remove saved token sequences so they are not used again
            extra_token_seqs_np = extra_token_seqs_np[num_needed_seq:]

            # add extra token_type_ids to hdf5 file
            curr_hdf5_file["token_type_ids"].resize(new_shape)
            curr_hdf5_file["token_type_ids"][-num_needed_seq:] = extra_ttid_seqs_np[
                :num_needed_seq
            ]
            # remove saved token_type_ids sequences so they are not used again
            extra_ttid_seqs_np = extra_ttid_seqs_np[num_needed_seq:]

    assert len(extra_token_seqs_np) == 0
    assert remainder == 0


if __name__ == "__main__":
    start_time = time.time()
    hdf5_files = glob(sys.argv[1] + "/train*")
    balance_hdf5_files(hdf5_files)
    total_time = time.time() - start_time
    print(f"total time to run re-balancing: {total_time}")
