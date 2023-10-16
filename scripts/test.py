import pandas as pd
import re, time
import numpy as np
from tabulate import tabulate

from main_llc import read_all_command, extract_outputs, evaluate
DEFAULT_SYSTEM_PROMPT = """\
Now I am providing you a command that a person can send to self-driving vehicle. \
Could you say whether this command needs to use how many of the below sections? \
It includes Perception, In-cabin monitoring, Localization, Vehicle control, Entertainmen, Personal data, Network access, Traffic laws. \
For example, I provide "Drive to the nearest train station.", Then it should include yes for Perception, Localization, Vehicle control, Network access.\
So you should output corresponding yes to these and no for others. Then you should output "1 0 1 1 0 0 1 0" to me. \
Now Tell me if the command is: """

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