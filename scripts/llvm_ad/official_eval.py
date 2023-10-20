import csv
import argparse

import os, sys, json
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from argparse import ArgumentParser
from utils.mics import extract_outputs
from utils.prompt import bc

def command_level_acc(true_dict, pred_dict) -> float:
    correct = 0
    total = 0
    for cid, true_labels in true_dict.items():
        if cid in pred_dict:
            pred_labels = pred_dict[cid]
            if len(true_labels) == len(pred_labels):
                correct += 1 if all([true_labels[i] == pred_labels[i] for i in range(len(true_labels))]) else 0
        total += 1
    acc = correct / total
    print(f"Command-level acc: {acc}")
    return acc


def question_level_acc(true_dict, pred_dict) -> float:
    correct = 0
    total = 0
    for cid, true_labels in true_dict.items():
        if cid in pred_dict:
            pred_labels = pred_dict[cid]
            if len(true_labels) == len(pred_labels):
                correct += sum([1 if true_labels[i] == pred_labels[i] else 0 for i in range(len(true_labels))])
        total += len(true_labels)
    acc = correct / total
    print(f"Question-level acc: {acc}")
    return acc


def open_true_csv(path="ucu.csv"):
    true_dict = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        fields = next(reader)
        for row in reader:
            command_id = int(row[0])
            labels = [1 if x == 'Yes' else 0 for x in row[2:]]
            true_dict[command_id] = labels
    return true_dict


def open_pred_csv(path):
    pred_dict = dict()
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            command_id = int(row[0])
            labels = [int(x) for x in row[1:]]
            pred_dict[command_id] = labels
    return pred_dict

def save_pred_csv(path):
    file_name = path.split('/')[-1].split('.')[0]
    save_folder = os.path.dirname(path)
    with open(path, "r") as f:
        data = json.load(f)
    with open(f"{save_folder}/{file_name}.csv", "w") as f:
        for result in data:
            command_id = result['id']
            pred = extract_outputs(result['response'], command_id)
            f.write(f"{command_id} {' '.join([str(x) for x in pred])}\n")
    return f"{save_folder}/{file_name}.csv"

if __name__ == "__main__":
    parser = ArgumentParser(description="Input two files to evaluate the accuracy of the model.")
    parser.add_argument("--ground_truth", "-g", type=str, default='/home/kin/workspace/llcommand/assets/ucu.csv', help='Ground truth file.')
    parser.add_argument("--evaluate_file", "-e", type=str, default='/home/kin/workspace/llcommand/assets/result/gpt-3.5-turbo-011_merged.json', help='Evaluate file json.')
    args = parser.parse_args()
    pred_csv = None
    if args.evaluate_file.endswith('.json'):
        pred_csv = save_pred_csv(args.evaluate_file)
        print("Since the input file is .json, we save the prediction to .csv file:\n", pred_csv, "\n")
    elif args.evaluate_file.endswith('.csv'):
        pred_csv = args.evaluate_file
    else:
        print("Wrong file format, please use .json")
        exit()
    print("Following is the evaluation result in official way: ")
    print(bc.BOLD)
    if pred_csv is not None and os.path.exists(pred_csv):
        tr_dict = open_true_csv(args.ground_truth)
        pr_dict = open_pred_csv(pred_csv)
        command_level_acc(tr_dict, pr_dict)
        question_level_acc(tr_dict, pr_dict)
    else:
        print("No such file, please use .json")
    print(bc.ENDC)