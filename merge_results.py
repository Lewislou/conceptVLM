import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def read_jsonl(file_path):    
    data = []    
    with open(file_path, 'r') as file:        
        for line in file:            
            obj = json.loads(line)            
            data.append(obj)    
    return data
def merge_json_files_in_order(file1, file2, output_file):
    # Load data from both files
    data1 = load_json(file1)
    data2 = read_jsonl(file2)

    # Check if both files have the same length
    if len(data1) != len(data2):
        raise ValueError("The two JSON files do not have the same number of entries.")

    # Merge data based on their order
    merged_data = []
    for item1, item2 in zip(data1, data2):
        merged_entry = {
            "question_id": item1['question_id'],
            "answer": item1['answer'],
            "gt_ans": item2['answer'],
            "image": item2['image'],
            "question": item2['question'],
            "answer_type":item2['answer_type'],
        }
        merged_data.append(merged_entry)

    # Save the merged data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, ensure_ascii=False, indent=4)

# Paths to the input files and the output file
file1_path = '/home/louw/InternVL/internvl_chat/results/textvqa_val_241203120445.json'  # Replace with the actual path to your first file
file2_path = '/home/louw/VLM/preprocess/datasets/11_28_v5_balance/eye_llava_instruct_clinical_val_v5_merge.jsonl'  # Replace with the actual path to your second file
output_file_path = 'internvl_v2_8B_val_v5_cot_choice.json'  # Output path

# Merge the files
merge_json_files_in_order(file1_path, file2_path, output_file_path)