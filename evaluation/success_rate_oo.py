import os
import json
import fnmatch
import argparse
from dotenv import load_dotenv
from gemini import gemini_evaluation

def process_scores(root_folder):
    # Load API key
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")

    for dirpath, dirnames, filenames in os.walk(root_folder): 
        eval_object = None
        
        # Load the evaluation object from the JSON file
        for filename in filenames:
            if fnmatch.fnmatch(filename, 'sentence*.json'): 
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f: 
                    data = json.load(f) 
                    eval_object = data.get('oo')
                    break  # Use the first match

        # Filter PNG files
        image_filenames = [filename for filename in filenames if filename.lower().endswith('.png')]
        success_rate = {}

        # Evaluate each image
        for filename in image_filenames: 
            path_image = os.path.join(dirpath, filename) 
            with open(path_image, 'rb') as f: 
                image_bytes = f.read() 
                response = gemini_evaluation(image_bytes, eval_object, gemini_key)
                success_rate[filename] = response 

        # Save results
        if success_rate: 
            output_path = os.path.join(dirpath, "success_rate_oo.json")
            with open(output_path, "w", encoding='utf-8') as file: 
                json.dump(success_rate, file, ensure_ascii=False, indent=4)
            print(f"Saved success rates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images using Gemini API and evaluation OO object.")
    parser.add_argument("--root", type=str, default="top3_log", help="Root folder to search in.")
    
    args = parser.parse_args()
    process_scores(args.root)
