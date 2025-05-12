import random 
import numpy as np 
from utils import get_text_embeds_without_uncond, cos_embedding_text

def init_pool(char_list, length, generation_scale):
    pool = [] 
    for _ in range(generation_scale) : 
        pool.append(''.join(random.sample(char_list, length))) 
    return pool 


def generate_offspring(p1: str, p2: str, char_list: list): 
    assert len(p1) == len(p2), "Two strings have to be the same length"
    p1_list = list(p1)
    p2_list = list(p2)
    # Cross
    cross_loc = random.randint(0, len(p1) - 1)
    p1_list[0:cross_loc] = p2[0:cross_loc]
    p2_list[0:cross_loc] = p1[0:cross_loc]
    # Mutation
    vari_loc = random.randint(0, len(p1) - 1)
    vari_char = random.choice(char_list)
    p1_list[vari_loc] = vari_char
    p2_list[vari_loc] = vari_char

    p1_offspring = ''.join(p1_list)
    p2_offspring = ''.join(p2_list)

    return p1_offspring, p2_offspring


def select(target_embedding, sentence, pool, score_dict, tokenizer, text_encoder, tour_size, mask):  
    pool_score = [] 
    # Compute fitness for pool
    for candidate in pool: 
        if candidate in score_dict.keys(): 
            temp_score = score_dict[candidate] 
            pool_score.append((temp_score,candidate)) 
            continue
        adv_prompt = sentence + ' ' + candidate
        temp_score = cos_embedding_text(target_embedding, adv_prompt, tokenizer=tokenizer, text_encoder=text_encoder,mask=mask)
        score_dict[candidate] = temp_score 
        pool_score.append((temp_score,candidate))
    # Tournament selection
    selected_pool = []
    random.shuffle(pool_score) 
    for i in range(0, len(pool_score), tour_size): 
        sub_pool = pool_score[i:i+tour_size] 
        selected_pool.append(max(sub_pool, key=lambda x: x[0])[1])
    if tour_size == 4 : 
        random.shuffle(pool_score) 
        for i in range(0, len(pool_score), tour_size): 
            sub_pool = pool_score[i:i+tour_size] 
            selected_pool.append(max(sub_pool, key=lambda x: x[0])[1])

    return selected_pool, pool_score
    

def evolution_strategy(target_sentence, sentence, char_list, length, generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask = None,seed=22520465) : 
    np.random.seed(seed)
    random.seed(seed)
    generation_list = init_pool(char_list, length, generation_scale)
    score_dict = {} 
    pool_score_log = []
    target_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)

    for _ in range(generation_num): 
        tem_pool = generation_list 
        indices = np.arange(len(generation_list)) 
        random.shuffle(indices) 
        # Select two random parents for crossover
        for i in range(0, len(generation_list), 2):
            candidate = generation_list[indices[i]] 
            mate = generation_list[indices[i+1]] 
            g1, g2 = generate_offspring(candidate, mate, char_list)
            tem_pool.append(g1)
            tem_pool.append(g2)
        
        generation_list, pool_score = select(target_embedding = target_embedding, sentence=sentence, pool=tem_pool, score_dict=score_dict, tokenizer=tokenizer, text_encoder=text_encoder, tour_size=tour_size, mask=mask)
        pool_score_log.append(pool_score)


    return score_dict,pool_score_log
