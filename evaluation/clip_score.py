import argparse
import os
import fnmatch
import json
import torch
from PIL import Image
import open_clip
from clip import CLIP_score 

def process_scores(root_folder: str, model_name: str, pretrained: str):
    # Load model, tokenizer, and transforms
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tar_sentence = None
    for dirpath, dirnames, filenames in os.walk(root_folder):

        # Tìm file JSON chứa tar_sentence
        for filename in filenames:
            if fnmatch.fnmatch(filename, 'sentence*.json'):
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    tar_sentence = data.get('tar_sentence')
                break 

        # Lọc file ảnh PNG
        image_filenames = [f for f in filenames if f.lower().endswith('.png')]
        clip_score_dict = {}

        for filename in image_filenames:
            path_image = os.path.join(dirpath, filename)
            image = preprocess(Image.open(path_image)).unsqueeze(0).to(device)
            score = CLIP_score(image, tar_sentence, model, tokenizer, device)
            clip_score_dict[filename] = score

        # Ghi kết quả ra file JSON
        if clip_score_dict:
            output_path = os.path.join(dirpath, "clip_score.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clip_score_dict, f, ensure_ascii=False, indent=4)
            print(f"[INFO] Saved CLIP scores to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images using OpenCLIP and compute CLIP scores.")
    parser.add_argument("--root", type=str, default="top3_log", help="Root folder to search in.")
    parser.add_argument("--model_name", type=str, default="ViT-g-14", help="Model name for OpenCLIP.")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="Pretrained checkpoint for OpenCLIP.")

    args = parser.parse_args()
    process_scores(args.root, args.model_name, args.pretrained)
