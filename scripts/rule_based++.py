"""
most of the codes generated by GPT-4 with plug in Advanced Data Analysis.
Compared to rule_based.py, this script solves letter capitalization, and add some new key words.
This is the method we reported in the paper called "rule-based" method.
"""
import pandas as pd
import os

# Load the CSV file into a DataFrame
commands_df = pd.read_csv("assets/ucu.csv", encoding='ISO-8859-1')
commands_df.head()


def evaluate_command(command):
    """
    This function evaluates a command and determines which sections it utilizes.
    It returns a dictionary with the sections marked as 'yes' or 'no'.
    """
    
    # Dictionary to store the evaluation results
    result = {
        "Perception": "no",
        "In-cabin monitoring": "no",
        "Localization": "no",
        "Vehicle control": "no",
        "Entertainment": "no",
        "Personal data": "no",
        "Network access": "no",
        "Traffic laws": "no"
    }
    
    # Evaluate the command for each section
    if any(word in command.lower() for word in ["drive", "change lanes", "route", "turn", "stop", "park"]):
        result["Perception"] = "yes"
        result["Vehicle control"] = "yes"
        result["Localization"] = "yes"
        result["Traffic laws"] = "yes"
    
    if any(word in command.lower() for word in ["video call", "play music", "watch movie", "movie", "music"]):
        result["Entertainment"] = "yes"
        result["Network access"] = "yes"
    
    if any(word in command.lower() for word in ["in-car system", "lights", "light"]):
        result["In-cabin monitoring"] = "yes"
    
    if any(word in command.lower() for word in ["report", "location", "nearest"]):
        result["Localization"] = "yes"
        result["Network access"] = "yes"

    if "call" in command.lower():
        result["Personal data"] = "yes"
    
    return result

# Apply the evaluation function to each command in the DataFrame
for index, row in commands_df.iterrows():
    command_result = evaluate_command(row["Command"])
    for key, value in command_result.items():
        commands_df.at[index, key] = value

commands_df.head()

# Convert the section columns to string data type
section_columns = [
    "Perception", "In-cabin monitoring", "Localization", "Vehicle control",
    "Entertainment", "Personal data", "Network access", "Traffic laws"
]

for column in section_columns:
    commands_df[column] = commands_df[column].astype(str)

# Apply the evaluation function to each command in the DataFrame again
for index, row in commands_df.iterrows():
    command_result = evaluate_command(row["Command"])
    for key, value in command_result.items():
        commands_df.at[index, key] = value

commands_df.head()

# Save the updated DataFrame to a CSV file
filename_without_extension = os.path.splitext(os.path.basename(__file__))[0]
output_path = f"code/llc/assets/result/rule_based/{filename_without_extension}_commands.csv"
commands_df.to_csv(output_path, index=False)
