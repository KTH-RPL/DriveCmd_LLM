# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama
import pandas as pd
from prompt import *

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
    ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-13b-Instruct",
    tokenizer_path: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-13b-Instruct/tokenizer.model",
    csv_path: str = "assets/ucu.csv",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512, # if the sentence is really long, should consider longer this one.
    max_batch_size: int = 6, # 
    max_gen_len: Optional[int] = None,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Model we use: ", ckpt_dir.split("/")[-1])

    commands, tasks = read_all_command(csv_path)
    print("Read all commands....")

    instructions = []
    for i, command in enumerate(commands):
        instructions.append([{"role": "system", "content": FORMAT_PROMPT}, 
                             {"role": "user", "content": DEFAULT_SYSTEM_PROMPT + command + OUTPUT_PROMPT}])
        if i>3: # debugging now
            break

    print("Start generating....")
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for i, (instruction, result) in enumerate(zip(instructions, results)):
        print(f"Asking command: {commands[i]} \n", "-"*20)
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    # TODO parse output to result and evaluate them.
if __name__ == "__main__":
    fire.Fire(main)
