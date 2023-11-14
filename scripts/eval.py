import os, sys, json
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from argparse import ArgumentParser
from utils.mics import read_all_command, print_result, extract_outputs
from utils.prompt import bc
from utils.official_eval import command_level_acc, question_level_acc, open_true_csv, open_pred_csv
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser(description="Input two files to evaluate the accuracy of the model.")
    parser.add_argument("--ground_truth", "-g", type=str, default='assets/ucu.csv', help='Ground truth file.')
    parser.add_argument("--evaluate_file", "-e", type=str, default='assets/result/gpt-4_best.json', help='Evaluate file, could be .csv or .npy')
    parser.add_argument('--official', '-o', action='store_true', help='Official.')
    args = parser.parse_args()
    temp, tasks, gt = read_all_command(args.ground_truth)
    (command_ids, _ ) = zip(*list(temp))
    all_pred = np.ones_like(gt)*(-1)
    if args.evaluate_file.endswith('.npy'):
        all_pred = np.load(args.evaluate_file)
    elif args.evaluate_file.endswith('.csv') and not args.official:
        commands, tasks, all_pred = read_all_command(args.evaluate_file)
    elif args.evaluate_file.endswith('.csv') and args.official:
        tr_dict = open_true_csv(args.ground_truth)
        pr_dict = open_pred_csv(args.evaluate_file)
    elif args.evaluate_file.endswith('.json'):
        with open(args.evaluate_file, "r") as f:
            data = json.load(f)
        for i, result in enumerate(data):
            command_id = result['id']
            pred = extract_outputs(result['response'], command_id)
            index_id = command_ids.index(command_id)
            all_pred[index_id,:] = pred # since command_id start with 1
    else:
        print("Wrong file format, please use .csv or .npy")
        exit()

    print(f"Evaluating {args.evaluate_file} ...")
    if args.official:
        print("Following is the evaluation result in official way:")
        print(bc.BOLD)
        command_level_acc(tr_dict, pr_dict)
        question_level_acc(tr_dict, pr_dict)
        print(bc.ENDC)
    else:
        error_num = np.sum(all_pred[:,0]==-1)
        if error_num > 0:
            print(f"\nTotal No correct output command Number: {bc.FAIL}{error_num}{bc.ENDC}, \
                \nError Ratio w Total: {bc.FAIL}{error_num/all_pred.shape[0]:.2f}{bc.ENDC}\n")
        print_result(all_pred, gt, tasks)