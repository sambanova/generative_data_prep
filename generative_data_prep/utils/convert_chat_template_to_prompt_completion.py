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
"""

import json
from argparse import ArgumentParser


def convert_chat_template_to_prompt_completion(
    input_path: str,
    output_prompt_completion_path: str,
    role_keyword: str,
    content_keyword: str,
    user_keyword: str,
    assistant_keyword: str,
):
    """Convert the jsonl file in input_path from chat template format to a jsonl of prompt completion pairs.

    Args:
        input_path: Path to input jsonl file in chat template format
        output_prompt_completion_path: Path to output file that will be in prompt completion format
        role_keyword: String keyword that defines the role of the chat turn.
        content_keyword: String keyword that defines where the text content is under.
        user_keyword: String keyword that defines if the turn is from a user. The value under "role_keyword"
        assistant_keyword:  String keyword that defines if the turn is from an assitant. The value under "role_keyword"
    """
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_prompt_completion_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            chat_turns = json.loads(line)
            prompt_completions = []
            curr_turn = {"prompt": "", "completion": ""}
            for turn in chat_turns:
                if turn[role_keyword] == user_keyword:
                    curr_turn["prompt"] = turn[content_keyword]
                elif turn[role_keyword] == assistant_keyword:
                    curr_turn["completion"] = turn[content_keyword]
                    prompt_completions.append(curr_turn)
                    curr_turn = {"prompt": "", "completion": ""}
            outfile.write(json.dumps(prompt_completions) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_prompt_completion_path", required=True, type=str)
    parser.add_argument("--role_keyword", default="role", type=str, required=False)
    parser.add_argument("--content_keyword", default="content", type=str, required=False)
    parser.add_argument("--user_keyword", default="user", type=str, required=False)
    parser.add_argument("--assistant_keyword", default="assistant", type=str, required=False)
    args = parser.parse_args()
    convert_chat_template_to_prompt_completion(
        args.input_path,
        args.output_prompt_completion_path,
        args.role_keyword,
        args.content_keyword,
        args.user_keyword,
        args.assistant_keyword,
    )
