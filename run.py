import numpy as np 
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
import torch 
import ast
from src import get_char_table, avoidance_strategy, compare_sentences, beam_search, evolution_strategy, auto_tune_threshold


if __name__ == "__main__":  
    parser = argparse.ArgumentParser() 
    parser.add_argument("--algorithm", type=str, default="genetic", choices=["genetic", "beam", "random"], help="The algorithm is implemented")
    parser.add_argument("--objective", type=str, default="te", choices=["te", "note"])
    parser.add_argument("--adv_length", type=int, default=5, help="The length of adversarial")
    parser.add_argument("--ori_sen", type=str, default="a cat")
    parser.add_argument("--tar_sen", type=str, default="a cat and a book") 
    parser.add_argument("--to", type=str, default="book")
    parser.add_argument("--space_limit", type=int, default=6000)
    parser.add_argument("--cosine_limit", type=float, default=0.35)
    parser.add_argument("--top_k", type=str, default="[]", help="if use beam, you need to define this such as [60,140,140,140,3], note that len(top_k) == adv_length") 
    parser.add_argument("--generation_num", type=int, default=50, help="the number of generation for genetic algorithm")
    parser.add_argument("--generation_scale", type=int, default=1280, help="The number of individuals for genetic algorithm") 
    parser.add_argument("--seed", type=int, default=22520465) 
    parser.add_argument("--num_repeat", type=int, default=3, help="Number of times to repeat the workflow, increasing the seed by 1 each time. For example, if seed=22520465 and num_repeat=3, the seed will be 22520465, 22520466, and 22520467")
    parser.add_argument("--more_words", type=str, default="[]", help="Do you want to add more words to search?")
    parser.add_argument("--num_cans", type=int, default=3)


    args = parser.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using {device}")

    # load model
    print(f"-----------Load model--------------")
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)

    # hyperparameters 
    algorithm = args.algorithm
    objective = args.objective
    adv_len = args.adv_length
    ori_sentence = args.ori_sen
    tar_sentence = args.tar_sen 
    to = args.to
    space_limit = args.space_limit 
    cosine_limit = args.cosine_limit
    top_k = ast.literal_eval(args.top_k)
    generation_num = args.generation_num 
    generation_scale = args.generation_scale 
    seed = args.seed
    num_repeat = args.num_repeat 
    more_words = ast.literal_eval(args.more_words)
    num_cans = args.num_cans

    # get char_list
    char_list = get_char_table()
    char_list = avoidance_strategy(char_list, to, tokenizer, text_encoder)
    char_list += more_words
    char_list = list(set(char_list))
    print(f"Character list: {char_list}")

    # implement 
    top_candidates = []
    cosin = compare_sentences(tar_sentence, ori_sentence, mask=None, tokenizer=tokenizer, text_encoder=text_encoder)
    print(f"Original cosine similarity between original sentence and target sentence: {cosin}")
    for i in range(num_repeat): 
        seed += 1
        print(f"Being repeat {i+1}")

        if objective == "note": 
            print(f"Use NoTE objectve")
            if algorithm == "beam": 
                print(f"Implement beam search with: {[tar_sentence, ori_sentence, adv_len, top_k]}")
                score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, adv_len, None, tokenizer, text_encoder, top_k)
            elif algorithm =="genetic": 
                print(f"Implement genetic algorithm with: {[tar_sentence, ori_sentence, adv_len, generation_num, generation_scale, seed]}")
                score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, adv_len, generation_num, generation_scale, tokenizer, text_encoder, 4, mask=None, seed=int(seed))

        elif objective == "te": 
            print(f"Use TE objectve")
            te, cosin = auto_tune_threshold(tar_sentence, ori_sentence, tokenizer, text_encoder, space_limit = space_limit, cosine_limit=cosine_limit)
            if algorithm == "beam": 
                print(f"Implement beam search with: {[tar_sentence, ori_sentence, adv_len, top_k]}")
                score_dict, pool_score_log = beam_search(tar_sentence, ori_sentence, char_list, adv_len, te, tokenizer, text_encoder, top_k)
            elif algorithm =="genetic": 
                print(f"Implement genetic algorithm with: {[tar_sentence, ori_sentence, adv_len, generation_num, generation_scale, seed]}")
                score_dict, pool_score_log = evolution_strategy(tar_sentence, ori_sentence, char_list, adv_len, generation_num, generation_scale, tokenizer, text_encoder, 4, mask=te, seed=int(seed))

        sorted_score= sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:num_cans]
        print(sorted_score)





