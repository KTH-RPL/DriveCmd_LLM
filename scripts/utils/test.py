import pandas as pd
import re, time
import numpy as np
from tabulate import tabulate
import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from main_llama import read_all_command, extract_outputs, evaluate

if __name__ == "__main__":
    commands, tasks, gt_array = read_all_command("/home/kin/workspace/llcommand/assets/ucu.csv")
    instructions = []
    with open("/home/kin/workspace/llcommand/output_content.txt", "r") as f:
        for line in f.readlines():
            instructions.append(line.strip())
    all_pred = []
    for result in instructions:
        pred = extract_outputs(result)
        # if pred is None: TODO, save i then rerun the command again.
        all_pred.append(pred)
    # print("Saving results....")
    all_pred = np.vstack(all_pred)
    # np.save("pred_res.npy", all_pred)
    acc = evaluate(all_pred, gt_array[:len(all_pred)])
    printed_data = []

    for i, task in enumerate(tasks):
        printed_data.append([task, acc[i]])
    printed_data.append(["Overall", np.mean(acc)])
    print(tabulate(printed_data, headers=['Task', 'Accuracy'], tablefmt='orgtbl'))
    print("Done!")