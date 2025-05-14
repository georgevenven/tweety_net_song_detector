import json
import shutil
import os
from tqdm import tqdm

def copy_files_from_json(json_file, source_dir, target_dir):
    with open(json_file, 'r') as f:
        file_list = json.load(f)
    
    os.makedirs(target_dir, exist_ok=True)
    
    for file_name in tqdm(file_list, desc="Copying files"):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy2(source_path, target_path)
    
    print(f"Copied {len(file_list)} files to {target_dir}")

# Usage
json_file = '/media/george-vengrovski/disk2/training_bengalese_finch_detector/combined_spec/labeled_files.json'
source_dir = '/media/george-vengrovski/disk2/training_bengalese_finch_detector/combined_spec'
target_dir = '/media/george-vengrovski/disk2/training_bengalese_finch_detector/bf_labeled'

copy_files_from_json(json_file, source_dir, target_dir)
