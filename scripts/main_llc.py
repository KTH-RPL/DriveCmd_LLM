# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama
import pandas as pd
DEFAULT_SYSTEM_PROMPT = """\
Now I am providing you a command that a person can send to self-driving vehicle. \
Could you say whether this command needs to use how many of the below sections? \
It includes Perception, In-cabin monitoring, Localization, Vehicle control, Entertainmen, Personal data, Network access, Traffic laws. \
For example, I provide "Drive to the nearest train station.", Then it should include yes for Perception, Localization, Vehicle control, Network access.\
So you should output corresponding yes to these and no for others. Then you should output "1 0 1 1 0 0 1 0" to me. \
Now Tell me if the command is:"""

def read_all_command(path: str):
    commands_df = pd.read_csv(path, encoding='ISO-8859-1')
    preview = commands_df.head()
    # Detailed analysis of each command
    all_commands_only = []
    task_name = preview.columns.to_list()[2:]
    for index, row in commands_df.iterrows():
        command = row["Command"].lower()
        all_commands_only.append(command)
    return all_commands_only, task_name


def main(
    ckpt_dir: str = "model/CodeLlama-7b",
    tokenizer_path: str = "model/CodeLlama-7b/tokenizer.model",
    csv_path: str = "assets/ucu.csv",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    commands, tasks = read_all_command(csv_path)
    instructions = []
    for i, command in enumerate(commands):
        instructions.append([{"role": "user", "content": DEFAULT_SYSTEM_PROMPT + command}])
        if i>5: # debugging now
            break
        
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
