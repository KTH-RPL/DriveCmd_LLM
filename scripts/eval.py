import os, sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from main_llama import read_all_command, print_result

if __name__ == "__main__":
    commands, tasks, pred = read_all_command("/home/kin/workspace/llcommand/assets/result/inferred_commands.csv")
    _, _, gt = read_all_command("/home/kin/workspace/llcommand/assets/ucu.csv")
    print_result(pred, gt, tasks)