import os 
import json
import fnmatch
from dotenv import load_dotenv
from gemini import gemini_evaluation

# load api 
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") 

root_folder = "top3_log/sentence1" 
for dirpath, dirnames, filenames in os.walk(root_folder): 
    for filename in filenames:
        if fnmatch.fnmatch(filename, 'sentence*.json'): 
            with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f: 
                data = json.load(f) 
                eval_object = data['oo']
    filenames = [filename for filename in filenames if filename.lower().endswith('.png')]
    success_rate = {}
    for filename in filenames: 
        path_image = os.path.join(dirpath, filename) 
        with open(path_image, 'rb') as f: 
            image_bytes = f.read() 
            response = gemini_evaluation(image_bytes, eval_object, gemini_key)
            success_rate[filename] = response 
        with open(os.path.join(dirpath,"success_rate_oo.json"), "w", encoding='utf-8') as file: 
            json.dump(success_rate, file, ensure_ascii=False, indent=4)