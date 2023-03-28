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

import h5py
from transformers import AutoTokenizer
from argparse import ArgumentParser
from generative_data_prep.utils.constants import TokenTypeIds

SEP_STR = '=' * 80 + '\n'
ATTENTION_SEPARATOR = '-' * 30 + 'ATTENTION SEPARATOR' + '-' * 30 + '\n'

def print_tokens(tokenizer, curr_prompt, curr_completion, wf):
    if curr_prompt != []:
        wf.write(f'----PROMPT-----\n{tokenizer.decode(curr_prompt)}\n')
    if curr_completion != []:
        wf.write(f'----COMPLETION-----\n{tokenizer.decode(curr_completion)}\n')
    
def decode_hdf5(hdf5_file_path: str, output_decoded_file_path: str, pretrained_tokenizer: str):  
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    
    with h5py.File(hdf5_file_path, 'r') as file_h5:
        with open(output_decoded_file_path, 'w') as wf:
            num_seqs = len(file_h5["input_ids"])
            seq_length = len(file_h5["input_ids"][0])

            wf.write(SEP_STR)
            wf.write(f'Number of Sequences: {num_seqs}\n')
            wf.write(f'Sequence length: {seq_length}\n')
            wf.write(SEP_STR)

            for seq_i in range(num_seqs):
                wf.write(f'SEQUENCE {seq_i + 1}\n')
                wf.write(SEP_STR)
                input_ids = file_h5['input_ids'][seq_i]
                token_type_ids = file_h5['token_type_ids'][seq_i]
                curr_prompt = []
                curr_completion = []
                curr_state = token_type_ids[0]
                for i, token, id in zip(range(seq_length), input_ids, token_type_ids):
                    if id != curr_state:
                        if id == int(TokenTypeIds.SEP):
                            curr_completion.append(token)
                        print_tokens(tokenizer, curr_prompt, curr_completion, wf)
                        curr_prompt = []
                        curr_completion = []
                        if id == int(TokenTypeIds.SEP):
                            wf.write(ATTENTION_SEPARATOR)

                    if id == int(TokenTypeIds.PADDING):
                        wf.write(f'Padding tokens: {seq_length - i}\n')
                        break
                    if id == int(TokenTypeIds.COMPLETION):
                        curr_completion.append(token)
                    if id == int(TokenTypeIds.PROMPT):
                        curr_prompt.append(token)

                    curr_state = id

                
                print_tokens(tokenizer, curr_prompt, curr_completion, wf)
                curr_prompt = []
                curr_completion = []
                
                wf.write('\n\n')
                wf.write(SEP_STR)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hdf5_file_path', required=True, type=str)
    parser.add_argument('--output_decoded_file_path', default=None, type=str, required=True)
    parser.add_argument('--pretrained_tokenizer', default='gpt2', type=str, required=False)
    args = parser.parse_args()
    decode_hdf5(args.hdf5_file_path, args.output_decoded_file_path, args.pretrained_tokenizer)