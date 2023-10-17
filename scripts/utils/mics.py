import pandas as pd
import re, time, wandb, os, sys
import numpy as np
from tabulate import tabulate
from .prompt import bc

def output_result(BASE_DIR, all_results, all_outputs, model_name, gt_array, tasks, debug_len=-1):
    print("Here are results...")
    all_pred = []
    os.makedirs(f"{BASE_DIR}/assets/result", exist_ok=True)
    for i, result in enumerate(all_results):
        pred = extract_outputs(result, i)
        all_pred.append(pred)

    with open(f"{BASE_DIR}/assets/result/{model_name}.txt", "w") as f:
        f.write(f"""\n""".join([str(result) for result in all_outputs]))
    if debug_len == -1:
        wandb.save(f"{BASE_DIR}/assets/result/{model_name}.txt")
    print("Saving results....")

    all_pred = np.vstack(all_pred)
    np.save(f"{BASE_DIR}/assets/result/{model_name}.npy", all_pred)
    if debug_len == -1:
        wandb.save(f"{BASE_DIR}/assets/result/{model_name}.npy")
    acc = print_result(all_pred, gt_array[:len(all_pred)], tasks)
    print(f"""Model we use: {bc.OKCYAN}{model_name}{bc.ENDC}""")

    # save to wandb
    score = {'overall': np.mean(acc)}
    score.update(dict(zip(tasks, acc)))
    wandb.log({"acc": score})
    wandb.finish()

def wandb_log(provide_detailed_explain, provide_few_shots, step_by_step, model_name, debug_len, slurm_job_id):
    wandb.init(entity="hdmaptest", project="llc", 
        name=f"{slurm_job_id}-{model_name}",
        mode= ("online" if debug_len == -1 else "offline"),
        # mode= "online",
        config={
        "provide_detailed_explain": provide_detailed_explain,
        "provide_few_shots": provide_few_shots,
        "step_by_step": step_by_step,
        "model_name": model_name,
        },
    )

def create_save_name(model_base_name, provide_detailed_explain, provide_few_shots, step_by_step, debug_len):
    flags = [
        '1' if provide_detailed_explain else '0',
        '1' if provide_few_shots else '0',
        '1' if step_by_step else '0',
        '' if debug_len == -1 else '-debug'
    ]
    flag_str = ''.join(flags)
    return model_base_name + "-" + flag_str

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
    
    pattern = r"Output is //\[([0-1\s]+)\]//"
    matches = re.findall(pattern, str(text))

    if len(matches) == 0:
        pattern = r"//\[([0-1\s]+)\]//"
        matches = re.findall(pattern, str(text))
    if len(matches) == 0:
        pattern = r"would be //\[([0-1\s]+)\]//"
        matches = re.findall(pattern, str(text))
    if len(matches) == 0:
        print(f"No match found in {bc.FAIL}{text}{bc.ENDC}, Need check later at {i}.")
        return np.array([-1 for _ in range(8)])
    
    np_res = [np.fromstring(match, dtype=int, sep=' ') for match in matches][0]
    if np_res.shape[0] != 8:
        print(f"Wrong shape in {text}, Need check later at {i}.")
        return np.array([-1 for _ in range(8)])
    return np_res

def evaluate(pred, gt):
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
    return acc