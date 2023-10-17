"""
# Created: 2023-10-16 18:34
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# Only this code is licensed under the terms of the MIT license. All other references are subjected to their own licenses.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

from typing import Optional

import fire

from codellama import Llama
import pandas as pd

# custom
from utils.prompt import *
from utils.mics import *
import torch.distributed as dist
import re, time, wandb
import numpy as np
from tabulate import tabulate
import os, sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))

def get_completion_from_user_input(user_input, generator, max_gen_len, temperature, top_p, \
                                   provide_detailed_explain=False, provide_few_shots = False, step_by_step=False,
                                   emphasis=""):
    fix_system_message = system_message
    if step_by_step:
        fix_system_message = step_system_message

    messages =  [{'role':'system', 'content': fix_system_message},]

    if not step_by_step and provide_detailed_explain:
        messages =  [  {'role':'system', 'content': fix_system_message+assistant},]

    if provide_few_shots:
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_1})
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_2})

    messages.append({'role':'user', 'content': f"{delimiter}{user_input}{delimiter}{emphasis}{delimiter}"})
    response = generator.chat_completion(
        [messages],  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return response[0]['generation']['content'], response[0]

def main(
    ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-7b-Instruct",
    # ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-13b-Instruct",
    # ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-34b-Instruct",
    csv_path: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/llc/assets/ucu.csv",
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_seq_len: int = 4096, # if the sentence is really long, should consider longer this one.
    max_batch_size: int = 6, # TODO changed according to the memory
    max_gen_len: Optional[int] = None,
    debug_len: int = 10, # TODO! if it's really big may have problem with memory
    provide_detailed_explain: bool = False,
    provide_few_shots: bool = False,
    step_by_step: bool = False,
    slurm_job_id: str = "00000",
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=ckpt_dir+"/tokenizer.model",
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model_name = create_save_name(ckpt_dir.split("/")[-1], provide_detailed_explain, provide_few_shots, step_by_step, debug_len)
    rank = dist.get_rank()

    if rank == 0:
        wandb_log(provide_detailed_explain, provide_few_shots, step_by_step, model_name, debug_len, slurm_job_id)

    print(f"""Model we use: {bc.OKCYAN}{model_name}{bc.ENDC}""")

    commands, tasks, gt_array = read_all_command(csv_path)
    print("Read all commands....")

    all_outputs = []
    all_results = []
    for i, command in enumerate(commands):
        start_time = time.time()
        response, style_response = get_completion_from_user_input(command, generator, max_gen_len, temperature, top_p, \
                                                  provide_detailed_explain=provide_detailed_explain, provide_few_shots=provide_few_shots, step_by_step=step_by_step)
        if rank == 0:
            wandb.log({"cost (s)": time.time() - start_time})
        if (i % 100 == 0 and debug_len == -1) or (debug_len>0):
            print(f"\n===== command {bc.BOLD}{i}{bc.ENDC}: {commands[i]} =====================\n")
            print(f"> {response}")
        all_results.append(response)
        all_outputs.append(style_response)
        if i>debug_len and debug_len != -1: # debugging now
            break

    if rank == 0:
        output_result(BASE_DIR, all_results, all_outputs, model_name, gt_array, tasks, debug_len=debug_len)

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")
