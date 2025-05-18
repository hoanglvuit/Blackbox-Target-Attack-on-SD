import os
import json
import fnmatch
import argparse
from dotenv import load_dotenv
from gemini import gemini_evaluation

def process_scores(root_folder, gemini_key):
    for dirpath, dirnames, filenames in os.walk(root_folder): 
        
        # Load the evaluation object from the JSON file
        for filename in filenames:
            if fnmatch.fnmatch(filename, 'sentence*.json'): 
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f: 
                    data = json.load(f) 
                    eval_object = data.get('to')
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
            output_path = os.path.join(dirpath, "success_rate_to.json")
            with open(output_path, "w", encoding='utf-8') as file: 
                json.dump(success_rate, file, ensure_ascii=False, indent=4)
            print(f"Saved success rates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images using Gemini API and evaluation target object.")
    parser.add_argument("--root", type=str, default="top3_log", help="Root folder to search in.")
    parser.add_argument("--api", type=str, default=None, help="API for Gemini")

    args = parser.parse_args()
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if args.api != None: 
        gemini_key = args.api
    process_scores(args.root, gemini_key)

