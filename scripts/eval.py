"""
# Created: 2023-10-16 23:49
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# Only this code is licensed under the terms of the MIT license. All other references are subjected to their own licenses.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import os, sys, json
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from argparse import ArgumentParser
from utils.mics import read_all_command, print_result, extract_outputs
from utils.prompt import bc
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Input two files to evaluate the accuracy of the model.")
    parser.add_argument("--ground_truth", "-g", type=str, default='/home/kin/workspace/llcommand/assets/ucu.csv', help='Ground truth file.')
    parser.add_argument("--evaluate_file", "-e", type=str, default='/home/kin/workspace/llcommand/assets/result/test.json', help='Evaluate file, could be .csv or .npy')
    args = parser.parse_args()
    _, tasks, gt = read_all_command(args.ground_truth)
    all_pred = np.ones_like(gt)*(-1)
    if args.evaluate_file.endswith('.npy'):
        all_pred = np.load(args.evaluate_file)
    elif args.evaluate_file.endswith('.csv'):
        commands, tasks, all_pred = read_all_command(args.evaluate_file)
    elif args.evaluate_file.endswith('.json'):
        with open(args.evaluate_file, "r") as f:
            data = json.load(f)
        for result in data:
            command_id = result['id']
            pred = extract_outputs(result['response'], command_id)
            all_pred[(command_id-1),:] = pred # since command_id start with 1
    else:
        print("Wrong file format, please use .csv or .npy")
        exit()
    print(f"\nTotal No correct output command Number: {bc.FAIL}{np.sum(all_pred[:,0]==-1)}{bc.ENDC}, \
          \nError Ratio w Total: {bc.FAIL}{np.sum(all_pred[:,0]==-1)/all_pred.shape[0]:.2f}{bc.ENDC}\n")
    print_result(all_pred, gt, tasks)