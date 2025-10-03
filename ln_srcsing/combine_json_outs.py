import os
import json

def combine_jsons_in_subfolders(base_dir, combined_name="combined.json"):
    # iterate over each subdirectory in the base dir
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        if os.path.isdir(subdir_path):  # ensure it's a folder (loan folder)
            combined_data = {}
            
            # iterate over all JSON files in that loan folder
            for file in os.listdir(subdir_path):
                if file.endswith(".json") and file != combined_name:
                    file_path = os.path.join(subdir_path, file)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        # store under key = filename without extension
                        key = os.path.splitext(file)[0]
                        combined_data[key] = data
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            
            # save combined output in that loan folder
            if combined_data:  # only save if something was found
                combined_path = os.path.join(subdir_path, combined_name)
                with open(combined_path, "w") as f:
                    json.dump(combined_data, f, indent=2)
                print(f"✅ Combined JSON saved to {combined_path}")
            else:
                print(f"⚠️ No JSON files found in {subdir_path}")

# Example usage:
# combine_jsons_in_subfolders("base_inventory")
