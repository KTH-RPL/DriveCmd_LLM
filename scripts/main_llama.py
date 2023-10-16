"""
# Created: 2023-10-16 18:34
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# Only this code is licensed under the terms of the MIT license. All other references are subjected to their own licenses.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

from typing import Optional

import fire

from llama import Llama
import pandas as pd

# custom
from utils.prompt import *
import torch.distributed as dist
import re, time
import numpy as np
from tabulate import tabulate

def read_all_command(path: str):
    commands_df = pd.read_csv(path, encoding='ISO-8859-1')
    commands_df.replace({'Yes': 1, 'No': 0, 'yes':1, 'no': 0}, inplace=True)
    preview = commands_df.head()
    # Detailed analysis of each command

    all_commands_only = commands_df["Command"].str.lower().tolist()
    task_name = commands_df.columns[2:].tolist()
    gt_array = commands_df[task_name].values
    return all_commands_only, task_name, gt_array

def extract_outputs(text, i=-1):
    
    pattern = r"//The output is \[([0-1\s]+)\]//"
    matches = re.findall(pattern, str(text))

    if len(matches) == 0:
        pattern = r"The output is \[([0-1\s]+)\]"
        matches = re.findall(pattern, str(text))

    if len(matches) == 0:
        pattern = r"would be \[([0-1\s]+)\]"
        matches = re.findall(pattern, str(text))
    if len(matches) == 0:
        print(f"No match found in {text}, Need check later at {i}.")
        return np.array([-1 for _ in range(8)])
    
    np_res = [np.fromstring(match, dtype=int, sep=' ') for match in matches][0]
    if np_res.shape[0] != 8:
        print(f"Wrong shape in {text}, Need check later at {i}.")
        return np.array([-1 for _ in range(8)])
    return np_res

# accuracy gt_array and all_pred
def evaluate(pred, gt):
    # pred and gt are numpy arrays
    # pred: (num_samples, num_tasks)
    # gt: (num_samples, num_tasks)
    num_samples, num_tasks = gt.shape
    assert pred.shape[0] == num_samples
    assert pred.shape[1] == num_tasks

    acc = []
    for i in range(num_tasks):
        acc.append(np.mean((pred[:, i] == gt[:, i]) & (pred[:, i] != -1)))
    return acc

def print_result(pred, gt, tasks):
    acc = evaluate(pred, gt[:len(pred)])
    printed_data = []
    for i, task in enumerate(tasks):
        printed_data.append([task, acc[i]])
    printed_data.append(["Overall", np.mean(acc)])
    print(tabulate(printed_data, headers=['Task', 'Accuracy'], tablefmt='orgtbl'))

def get_completion_from_user_input(user_input, generator, provide_detailed_explain=False, provide_few_shots = False):
    messages =  [  
    {'role':'system', 'content': system_message},
    ]

    if provide_detailed_explain:
        messages.append({'role':'assistant', 'content': f"{delimiter}{assitant}{delimiter}"})

    if provide_few_shots:
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_1})
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_2})

    messages.append({'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"})

def main(
    ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-7b-Instruct",
    tokenizer_path: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-7b-Instruct/tokenizer.model",
    # ckpt_dir: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-13b-Instruct",
    # tokenizer_path: str = "/proj/berzelius-2023-154/users/x_qinzh/workspace/codellama/CodeLlama-13b-Instruct/tokenizer.model",
    csv_path: str = "assets/ucu.csv",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512, # if the sentence is really long, should consider longer this one.
    max_batch_size: int = 6, # TODO changed according to the memory
    max_gen_len: Optional[int] = None,
    debug_len: int = 10, # TODO! if it's really big may have problem with memory
    provide_detailed_explain: bool = False,
    provide_few_shots: bool = False,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max(max_batch_size, debug_len*2),
    )
    print(f"""Model we use: {bc.OKCYAN}{ckpt_dir.split("/")[-1]}{bc.ENDC}""")

    commands, tasks, gt_array = read_all_command(csv_path)
    print("Read all commands....")

    instructions = []
    for i, command in enumerate(commands):
        response = get_completion_from_user_input(command, provide_detailed_explain=provide_detailed_explain, provide_few_shots=provide_few_shots )
        if i>debug_len: # debugging now
            break

    # print("Start generating....")
    # results = generator.chat_completion(
    #     instructions,  # type: ignore
    #     max_gen_len=max_gen_len,
    #     temperature=temperature,
    #     top_p=top_p,
    # )

    # print("Here are results...")
    # all_results = []
    # all_pred = []
    # rank = dist.get_rank()
    # if rank == 0:
    #     for i, result in enumerate(results):
    #         print(f"\n===== command {bc.BOLD}{i}{bc.ENDC}: {commands[i]} =====================\n")
    #         print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
    #         all_results.append(result['generation'])

    #     with open("output_content.txt", "w") as f:
    #         # with '---' as separator
    #         f.write(f"""\n""".join([str(result) for result in all_results]))

    #     for i, result in enumerate(all_results):
    #         pred = extract_outputs(result, i)
    #         # if pred is None: TODO, save i then rerun the command again.
    #         all_pred.append(pred)

    # print("Saving results....")
    # if rank == 0:
    #     all_pred = np.vstack(all_pred)
    #     np.save("pred_res.npy", all_pred)
    #     print_result(all_pred, gt_array[:len(all_pred)], tasks)

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")
