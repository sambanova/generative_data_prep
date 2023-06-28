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

import os
import random
import shutil
import time

from tqdm import tqdm


def large_file_shuffle(
    input_file_path: str,
    output_dir: str,
    concat_splits: bool = True,
    num_splits: int = 150000,
) -> str:
    """Fast approximate shuffling for massive files. This function
    1. splits the input file into [num_splits]
    2. shuffles each split
    3. If yous set concat_splits=True, then the splits are concatenated in a random order

    Requirements:
    1. you must have enough storage in output_dir to store a shuffled file the same size as input_file_path
    2. You must be able to fit size(input_file_path)/num_splits into RAM

    Warning:
    This function does not do fair shuffling, but is a very close approximation.
    It uses round robin shuffling so it will in fact evenly distribute the input lines
    among the output.

    Args:
        input_file_path (str):
        output_dir (str):
        concat_splits (bool): flag that determines if the output should be concatenated or not
        num_splits (int): number of splits to split input file into,
        the more splits the slower it will be, but the better approximation
        to a fare shuffle

    Returns:
        output_file_path (str): The file path where the shuffled input was saved
    """
    print("PERFORMING LARGE FILE APPROXIMATE SHUFFLING")
    start_time = time.time()
    _, file_extension = os.path.splitext(input_file_path)
    split_dir = os.path.join(output_dir, "splits")
    if concat_splits:
        output_path = os.path.join(output_dir, "shuffled" + file_extension)
    else:
        output_path = split_dir

    if os.path.isdir(split_dir):
        print(
            f"WARNING - the split directory {split_dir} exists, if you do not manually abort this run in 5 seconds, it will be deleted and over-written"
        )
        time.sleep(5)
        shutil.rmtree(split_dir)
    os.mkdir(split_dir)

    if os.path.isfile(output_path):
        print(
            f"WARNING - the output file path {output_path} exists, if you do not manually abort this run in 5 seconds, it will be deleted and over-written"
        )
        time.sleep(5)
        os.remove(output_path)

    prev_time = time.time()
    print("splitting file")
    split_command = f"split -d -n r/{num_splits} {input_file_path} {split_dir}/"
    os.system(split_command)
    print(f"splitting took {time.time() - prev_time} seconds (used round robin splitting)")

    prev_time = time.time()
    print(f"shuffling {num_splits} files")
    file_list = list(os.listdir(split_dir))
    for file in tqdm(file_list):
        curr_file_path = os.path.join(split_dir, file)
        shuf_command = f"shuf {curr_file_path} --output={curr_file_path}"
        os.system(shuf_command)
    print(f"finished shuffling {num_splits} files. Took {time.time() - prev_time} seconds")

    if concat_splits:
        random_split_list = list(range(num_splits))
        random.shuffle(random_split_list)
        prev_time = time.time()
        print(f"concatenating shuffled splits")
        for rand_ind in tqdm(random_split_list):
            curr_file_path = os.path.join(split_dir, file_list[rand_ind])
            concat_command = f"cat {curr_file_path} >> {output_path}"
            os.system(concat_command)
            os.remove(curr_file_path)
        print(f"Finished concatenating files. Took {time.time() - prev_time} seconds")
        shutil.rmtree(split_dir)

    print(f"TOTAL TIME: {time.time() - start_time}")
    return output_path
