import os
import json
import argparse

def process_scores(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder): 
        success_rate = {}
        if 'success_rate_oo.json' in filenames and 'success_rate_to.json' in filenames: 
            with open(os.path.join(dirpath,'success_rate_oo.json'), 'r') as f1, open(os.path.join(dirpath,'success_rate_to.json'), 'r') as f2:
                sr1 = json.load(f1) 
                sr2 = json.load(f2) 
            success_rate = {key: sr1[key] & sr2[key] for key in sr1} 
            output_path = os.path.join(dirpath, 'success_rate_both.json')
            with open(output_path ,'w', encoding='utf-8') as f: 
                json.dump(success_rate, f, ensure_ascii=False, indent=4) 
            print(f"Saved success rates to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate if images contain both original object and target object")
    parser.add_argument("--root", type=str, default="output")

    args = parser.parse_args()
    process_scores(args.root)
