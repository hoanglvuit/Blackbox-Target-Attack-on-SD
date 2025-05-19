import os
import sys
import json
import torch
import argparse
from diffusers import StableDiffusionPipeline


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)
from src.utils import generate_images 


def save_images(images_list, dirpath):
    os.makedirs(dirpath, exist_ok=True)
    all_images = [img for row in images_list for img in row]
    for idx, image in enumerate(all_images, start=1):
        save_path = os.path.join(dirpath, f"{idx}.png")
        image.save(save_path, format='PNG')


def create_ori_target(root_folder):
    target_folder = os.path.join(root_folder, 'target')
    original_folder = os.path.join(root_folder, 'original')
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(original_folder, exist_ok=True)

    for file in os.listdir(root_folder):
        if file.endswith('.json') and os.path.isfile(os.path.join(root_folder, file)):
            file_path = os.path.join(root_folder, file)
            print(f"Processing: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tar_sentence = data.get('tar_sentence', '')
                ori_sentence = data.get('ori_sentence', '')
                oo = data.get('oo', '')

            with open(os.path.join(target_folder, 'target.json'), 'w', encoding='utf-8') as tf:
                json.dump({tar_sentence: 0}, tf, ensure_ascii=False, indent=2)

            with open(os.path.join(original_folder, 'original.json'), 'w', encoding='utf-8') as of:
                json.dump({ori_sentence: 0}, of, ensure_ascii=False, indent=2)

            return oo, tar_sentence
        

def create_image(root_folder, start_sentence):
    for dirpath, _, filenames in os.walk(root_folder):
        if os.path.abspath(dirpath) == os.path.abspath(root_folder):
            continue

        for file in filenames:
            file_path = os.path.join(dirpath, file)

            if file in ['original.json', 'target.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = list(data.keys())
                print(f"Generating images for {file}: {prompts}")
                images = generate_images(prompts, pipe, generator, num_image=10)
                save_images(images, dirpath)

            elif file == 'score_dict.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = [start_sentence + ' ' + key for key in data.keys()]
                print(f"Generating images for score_dict.json: {prompts}")
                images = generate_images(prompts, pipe, generator, num_image=10)
                save_images(images, dirpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from sentences in a root folder.")
    parser.add_argument("--root", type=str, default="top3_log/sentence1", help="Root folder to process.")
    parser.add_argument("--model", type=str, default="1.5", help="Stable Diffusion version")
    parser.add_argument("-seed", type=int, default=22520465)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    version = args.model 
    seed = args.seed 

    if version == "1.5":  
        pipe = StableDiffusionPipeline.from_pretrained(
            'sd-legacy/stable-diffusion-v1-5',
            torch_dtype=torch.float16,
            use_auth_token=True,
            safety_checker=None
        ).to(device)
    elif version == "1.4": 
        pipe = StableDiffusionPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4', 
            revision='fp16',
            torch_dtype=torch.float16, 
            safety_checker=None, 
            use_auth_token=True).to(device)
    generator = torch.Generator(device).manual_seed(seed)

    # implement
    root_path = args.root 
    oo, tar_sentence = create_ori_target(root_path) 
    start_sentence = tar_sentence[:tar_sentence.find(oo) + len(oo)].strip() 
    create_image(root_path, start_sentence)