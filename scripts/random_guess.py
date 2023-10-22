"""
many of the codes generated by GPT-4
"""
import pandas as pd
import os
import random

# Load the CSV file into a DataFrame
commands_df = pd.read_csv("code/llvm/data/ucu.csv", encoding='ISO-8859-1')

keys = ["Perception", "In-cabin monitoring", "Localization", "Vehicle control", 
            "Entertainment", "Personal data", "Network access", "Traffic laws"]

def random_assign():
    result = {key: random.choice(["yes", "no"]) for key in keys}
    return result

for column in keys:
    commands_df[column] = commands_df[column].astype(str)

# Apply the evaluation function to each command in the DataFrame again
for index, row in commands_df.iterrows():
    command_result = random_assign()
    for key, value in command_result.items():
        commands_df.at[index, key] = value

# Save the updated DataFrame to a CSV file
filename_without_extension = os.path.splitext(os.path.basename(__file__))[0]
output_path = f"code/llc/assets/result/rule_based/{filename_without_extension}_commands.csv"
commands_df.to_csv(output_path, index=False)
