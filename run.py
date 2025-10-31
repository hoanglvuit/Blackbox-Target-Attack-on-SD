import os
import json
import argparse
import numpy as np
import torch
import yaml
from pprint import pprint
from transformers import CLIPTextModel, CLIPTokenizer
from src import get_char_table, avoidance_strategy, compare_sentences, beam_search, evolution_strategy, auto_tune_threshold

def read_json(json_path: str): 
    """ 
    Read the json file and return the original sentence, target sentence, object of original sentence, and object of target sentence
    """
    with open (json_path, 'r') as file:
        input_data = json.load(file)
    
    ori_sen = input_data['ori_sentence']
    tar_sen = input_data['tar_sentence']
    oo = input_data['oo']
    to = input_data['to']
    
    return ori_sen, tar_sen, oo, to

def save_log(mode_dir, score_dict, pool_score_log):
    os.makedirs(mode_dir, exist_ok=True)
    with open(os.path.join(mode_dir, 'score_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(mode_dir, 'pool_score_log.json'), 'w', encoding='utf-8') as f:  
        json.dump(pool_score_log, f, ensure_ascii=False, indent=4)

def search(cfg, log_sentence_path: str, tar_sentence: str, ori_sentence: str, char_list: list, tokenizer, text_encoder, TE=True): 


    beam_configs = {
    "beam_uni": cfg['beam_uni_widths'],
    "beam_in": cfg['beam_in_widths'],
    "beam_de": cfg['beam_de_widths'],
    }

    seeds = np.arange(cfg['start_seed'], cfg['start_seed']+cfg['num_seeds'])
    generation_num = cfg['generation_num']
    generation_scale = cfg['generation_scale']
    tour_size = cfg['tour_size']

    if TE:
        mode_dir = os.path.join(log_sentence_path, 'TE')
        os.makedirs(mode_dir, exist_ok=True)
        te, cosin = auto_tune_threshold(tar_sentence, ori_sentence, tokenizer, text_encoder, space_limit = cfg['space_limit'], cosine_limit=cfg['cosine_limit'])
        with open(os.path.join(mode_dir, 'cosin.json'), 'w', encoding='utf-8') as f:
            json.dump(cosin, f, ensure_ascii=False, indent=4)
    else:
        mode_dir = os.path.join(log_sentence_path, 'NoTE')
        os.makedirs(mode_dir, exist_ok=True)
        cosin = compare_sentences(tar_sentence, ori_sentence, mask=None, tokenizer=tokenizer, text_encoder=text_encoder)
        with open(os.path.join(mode_dir, 'cosin.json'), 'w', encoding='utf-8') as f:
            json.dump(cosin, f, ensure_ascii=False, indent=4)

    
    beam_configs = {
        "beam_uni": cfg['beam_uni_widths'],
        "beam_in": cfg['beam_in_widths'],
        "beam_de": cfg['beam_de_widths'],
    }

    seeds = np.arange(cfg['start_seed'], cfg['start_seed']+cfg['num_seeds'])
    generation_num = cfg['generation_num']
    generation_scale = cfg['generation_scale']
    tour_size = cfg['tour_size']

    if cfg['beam_uni']:
        beam_uni_widths = cfg['beam_uni_widths']
        if TE:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], te, tokenizer, text_encoder, beam_uni_widths)
        else:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], None, tokenizer, text_encoder, beam_uni_widths)
        save_log(os.path.join(mode_dir, 'beam_uni'), score_dict, pool_score_log)

    if cfg['beam_in']:
        beam_in_widths = cfg['beam_in_widths']
        if TE:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], te, tokenizer, text_encoder, beam_in_widths)
        else:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], None, tokenizer, text_encoder, beam_in_widths)
        save_log(os.path.join(mode_dir, 'beam_in'), score_dict, pool_score_log)
    
    if cfg['beam_de']:
        beam_de_widths = cfg['beam_de_widths']
        if TE:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], te, tokenizer, text_encoder, beam_de_widths)
        else:
            score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], None, tokenizer, text_encoder, beam_de_widths)
        save_log(os.path.join(mode_dir, 'beam_de'), score_dict, pool_score_log)

    # Genetic algorithm
    if cfg['popop']:
        for seed in seeds:
            seed_dir = os.path.join(mode_dir, 'popop', str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            if TE:
                score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask=te, seed=int(seed))
            else:
                score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, cfg['len_prompt'], generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask=None, seed=int(seed))
            save_log(os.path.join(seed_dir), score_dict, pool_score_log)

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def print_config(cfg):
    print("=" * 60)
    print("🔧 CONFIGURATION")
    print("=" * 60)
    pprint(cfg, sort_dicts=False)
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with YAML config file.")
    parser.add_argument("--config", type=str, required=False, default="config.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_config(cfg)



    # load model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    tokenizer = CLIPTokenizer.from_pretrained(cfg['model'])
    text_encoder = CLIPTextModel.from_pretrained(cfg['model']).to(device)

    for file in os.listdir(cfg['dataset_path']):
        if file.endswith('.json'):
            json_path = os.path.join(cfg['dataset_path'], file)
            ori_sen, tar_sen, oo, to = read_json(json_path)
            print(f"First sentence: {ori_sen}")
            print(f"Second sentence: {tar_sen}")
            print(f"Original object: {oo}")
            print(f"Target object: {to}") 
        

        
        # avoidance strategy
        char_list = get_char_table()
        char_list = avoidance_strategy(char_list, to, tokenizer, text_encoder)

        log_sentence_path = os.path.join(cfg['log_path'], file.replace('.json', ''))
        os.makedirs(log_sentence_path, exist_ok=True)
        search(cfg, log_sentence_path, tar_sen, ori_sen, char_list, tokenizer, text_encoder, TE=True)
        search(cfg, log_sentence_path, tar_sen, ori_sen, char_list, tokenizer, text_encoder, TE=False)