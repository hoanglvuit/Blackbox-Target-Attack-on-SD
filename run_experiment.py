import os
import json
import argparse
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from src import get_char_table, avoidance_strategy, compare_sentences, beam_search, evolution_strategy, auto_tune_threshold

def main(args):
    sentence_name = args.sentence  
    sentence_path = f'dataset/{sentence_name}.json'
    log_dir = f'log/{sentence_name}'
    space_limit = args.space_limit 
    cosine_limit = args.cosine_limit 

    # Step 1: Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Step 2: Load and copy input data
    with open(sentence_path, 'r') as file:
        data = json.load(file)
    with open(f'{log_dir}/{sentence_name}.json', 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    oo = data['oo']
    to = data['to']
    ori_sentence = data['ori_sentence']
    tar_sentence = data['tar_sentence']
    print('Original object:', oo)
    print('Target object:', to)
    print('Original sentence:', ori_sentence)
    print('Target sentence:', tar_sentence)

    # Step 3: Load tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to('cuda')
    len_prompt = 5

    # Step 4: Avoidance strategy
    char_list = get_char_table()
    char_list = avoidance_strategy(char_list, to, tokenizer, text_encoder)

    # Step 5: NoTE mode
    note_dir = f"{log_dir}/NoTE"
    os.makedirs(note_dir, exist_ok=True)
    cosin = compare_sentences(tar_sentence, ori_sentence, mask=None, tokenizer=tokenizer, text_encoder=text_encoder)
    with open(f'{note_dir}/cosin.json', 'w', encoding='utf-8') as f:
        json.dump(cosin, f, ensure_ascii=False, indent=4)

    # Beam search variants
    beam_configs = {
        "beam_uni": [60,140,140,140,3],
        "beam_in": [60,20,50,350,3],
        "beam_de": [60,350,50,20,3],
    }
    for name, widths in beam_configs.items():
        sub_dir = f"{note_dir}/{name}"
        os.makedirs(sub_dir, exist_ok=True)
        score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, len_prompt, None, tokenizer, text_encoder, widths)
        with open(f'{sub_dir}/score_dict.json', 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)
        with open(f'{sub_dir}/pool_score_log.json', 'w', encoding='utf-8') as f:
            json.dump(pool_score_log, f, ensure_ascii=False, indent=4)

    # POPOP (NoTE)
    popop_dir = f"{note_dir}/popop"
    os.makedirs(popop_dir, exist_ok=True)
    generation_num = 50
    generation_scale = 1280
    tour_size = 4
    seeds = np.arange(22520465, 22520465+10)
    for seed in seeds:
        seed_dir = f"{popop_dir}/{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, len_prompt, generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask=None, seed=int(seed))
        with open(f'{seed_dir}/score_dict.json', 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)
        with open(f'{seed_dir}/pool_score_log.json', 'w', encoding='utf-8') as f:
            json.dump(pool_score_log, f, ensure_ascii=False, indent=4)

    # TE mode
    te_dir = f"{log_dir}/TE"
    os.makedirs(te_dir, exist_ok=True)
    te, cosin = auto_tune_threshold(tar_sentence, ori_sentence, tokenizer, text_encoder, space_limit = space_limit, cosine_limit=cosine_limit)
    with open(f'{te_dir}/cosin.json', 'w', encoding='utf-8') as f:
        json.dump(cosin, f, ensure_ascii=False, indent=4)

    # Beam search TE
    for name, widths in beam_configs.items():
        sub_dir = f"{te_dir}/{name}"
        os.makedirs(sub_dir, exist_ok=True)
        score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, len_prompt, te, tokenizer, text_encoder, widths)
        with open(f'{sub_dir}/score_dict.json', 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)
        with open(f'{sub_dir}/pool_score_log.json', 'w', encoding='utf-8') as f:
            json.dump(pool_score_log, f, ensure_ascii=False, indent=4)

    # POPOP (TE)
    popop_dir = f"{te_dir}/popop"
    os.makedirs(popop_dir, exist_ok=True)
    for seed in seeds:
        seed_dir = f"{popop_dir}/{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, len_prompt, generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask=te, seed=int(seed))
        with open(f'{seed_dir}/score_dict.json', 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)
        with open(f'{seed_dir}/pool_score_log.json', 'w', encoding='utf-8') as f:
            json.dump(pool_score_log, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True, help="Name of the sentence file, e.g. sentence14")
    parser.add_argument("--space_limit", type=int, default=6000)
    parser.add_argument("--cosine_limit", type=str, default=0.35)
    args = parser.parse_args()
    main(args)
