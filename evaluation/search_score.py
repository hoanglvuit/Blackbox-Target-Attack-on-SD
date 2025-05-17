import os
import json
import argparse

def process_scores(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.abspath(dirpath) == os.path.abspath(root_folder):
            continue
        
        if 'cosin.json' in filenames and 'score_dict.json' in filenames:
            cosin_path = os.path.join(dirpath, 'cosin.json')
            score_dict_path = os.path.join(dirpath, 'score_dict.json')
            
            with open(cosin_path, 'r', encoding='utf-8') as file:
                base_score = json.load(file)

            if not isinstance(base_score, (int, float)):
                print(f"[WARNING] Skipped {dirpath}: 'cosin.json' must contain a single number.")
                continue

            with open(score_dict_path, 'r', encoding='utf-8') as file:
                scores = json.load(file)
                score_list = list(scores.values())
                search_scores = [i - base_score for i in score_list]

            print(f"[INFO] Processed {dirpath}: {search_scores}")
            with open(os.path.join(dirpath, 'search_score.json'), 'w', encoding='utf-8') as f:
                json.dump(search_scores, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate search scores from cosin and score_dict JSON files.")
    parser.add_argument("--root", type=str, default="top3_log/sentence1", help="Root folder to search in.")
    
    args = parser.parse_args()
    process_scores(args.root)
