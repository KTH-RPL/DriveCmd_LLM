import openai
import os
from dotenv import load_dotenv, find_dotenv
import fire
from utils.prompt import *
from utils.mics import read_all_command, output_result, wandb_log, create_save_name, save_response_to_json
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))

def get_completion_from_user_input(user_input, provide_detailed_explain=False, provide_few_shots = False, step_by_step = False, model="gpt-3.5-turbo", temperature=0):
    fix_system_message = system_message
    if step_by_step:
        fix_system_message = step_system_message
        
    messages =  [  
    {'role':'system', 'content': fix_system_message},
    ]

    if not step_by_step and provide_detailed_explain:
        messages.append({'role':'assistant', 'content': f"{delimiter}{assistant}{delimiter}"})

    if provide_few_shots:
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_1})
        messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
        messages.append({'role':'assistant', 'content': few_shot_assistant_2})

    messages.append({'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # 控制模型输出的随机程度
    )
    return response.choices[0].message["content"]

def main(
    csv_path: str = "/proj/berzelius-2023-154/users/x_yiyan/code/llvm/data/ucu.csv",
    temperature: float = 0.0,
    provide_detailed_explain: bool = False,
    provide_few_shots: bool = False,
    step_by_step: bool = False,
    model: str = "gpt-3.5-turbo",
    debug_len: int = 10, # TODO! if it's really big may have problem with memory
    slurm_job_id: str = "00000",
):

    commands_w_id, tasks, gt_array = read_all_command(csv_path)
    print("1. Finished Read all commands!")
    wandb_log(provide_detailed_explain, provide_few_shots, step_by_step, model, debug_len, slurm_job_id)
    model_name = create_save_name(model, provide_detailed_explain, provide_few_shots, step_by_step, debug_len)

    all_results = []
    json_file_path = f"{BASE_DIR}/assets/result/{model_name}.json" # PLEASE DO NOT CHANGE THIS PATH
    numpy_file_path = f"{BASE_DIR}/assets/result/{model_name}.npy" # PLEASE DO NOT CHANGE THIS PATH
    os.makedirs(f"{BASE_DIR}/assets/result", exist_ok=True)

    for (i, command) in commands_w_id:
        start_time = time.time()

        response = get_completion_from_user_input(command, 
                                                    provide_detailed_explain=provide_detailed_explain, provide_few_shots=provide_few_shots, step_by_step=step_by_step, \
                                                    model=model, temperature=temperature)

        style_response = {'id': i, 'command': command, 'response': response}
        save_response_to_json(style_response, json_file_path)

        if (i % 100 == 0 and debug_len == -1) or (debug_len>0):
            print(f"\n===== command {bc.BOLD}{i}{bc.ENDC}: {command} =====================\n")
            print(f"> {response}")

        all_results.append(response)
        if i>debug_len and debug_len != -1: # debugging now
            break

    output_result(numpy_file_path, json_file_path, model_name, all_results, gt_array, tasks, debug_len=debug_len)

if __name__ == "__main__":
    start_time = time.time()

    # read environment variables from .env
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.getenv('OPENAI_API_KEY')

    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")