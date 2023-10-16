# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str = "model/CodeLlama-7b",
    tokenizer_path: str = "model/CodeLlama-7b/tokenizer.model",
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

    instructions = [
        # [
        #     {
        #         "role": "user",
        #         "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
        #     }
        # ],
        # [
        #     {
        #         "role": "user",
        #         "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
        #     }
        # ],
        [
            {
                "role": "system",
                "content": "Provide answers in Python",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of average of all contiguous sublists of a given list.",
            }
        ],
    ]
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
