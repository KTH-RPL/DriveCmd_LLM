"""
# Created: 2023-10-16 23:49
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# Only this code is licensed under the terms of the MIT license. All other references are subjected to their own licenses.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from argparse import ArgumentParser
from utils.mics import read_all_command, print_result
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Input two files to evaluate the accuracy of the model.")
    parser.add_argument("--ground_truth", "-g", type=str, default='/home/kin/workspace/llcommand/assets/ucu.csv', help='Ground truth file.')
    parser.add_argument("--evaluate_file", "-e", type=str, default='/home/kin/workspace/llcommand/result/inferred_commands.csv', help='Evaluate file, could be .csv or .npy')
    args = parser.parse_args()
    _, _, gt = read_all_command(args.ground_truth)
    if args.evaluate_file.endswith('.npy'):
        pred = np.load(args.evaluate_file)
    elif args.evaluate_file.endswith('.csv'):
        commands, tasks, pred = read_all_command(args.evaluate_file)
    else:
        print("Wrong file format, please use .csv or .npy")
    print_result(pred, gt, tasks)