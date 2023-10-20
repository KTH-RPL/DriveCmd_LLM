

file_list = [
"assets/result/gpt-3.5-turbo-011_v2_5357055.json",
"assets/result/gpt-3.5-turbo-011_v2_5357056.json",
"assets/result/gpt-3.5-turbo-011_v2_5357057.json",
"assets/result/gpt-3.5-turbo-011_v2_5357058.json",
"assets/result/gpt-3.5-turbo-011_v2_5357059.json",
"assets/result/gpt-3.5-turbo-011_v2_5357060.json",
"assets/result/gpt-3.5-turbo-011_v2_5357061.json",
"assets/result/gpt-3.5-turbo-011_v2_5357062.json",
"assets/result/gpt-3.5-turbo-011_v2_5357064.json",
"assets/result/gpt-3.5-turbo-011_v2_5357065.json",
"assets/result/gpt-3.5-turbo-011_v2_5357066.json"
]


import json

def merge_json_files(filenames, output_filename):
    all_data = []

    for filename in filenames:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                print(f"Warning: Data in {filename} is not a list.")
                all_data.append(data)

    with open(output_filename, 'w') as f:
        json.dump(all_data, f, indent=4)

# Example usage:
merge_json_files(file_list, "assets/result/gpt-3.5-turbo-011_v2_merged.json")
