import random 
import numpy as np 
from utils import get_text_embeds_without_uncond, cos_embedding_text

def init_pool(char_list, length, generation_scale):
    """
    Khởi tạo một quần thể (pool) gồm các chuỗi ngẫu nhiên.

    Args:
        char_list (list): Bảng kí tự đã được định nghĩa trước. 
        length (int): Độ dài mỗi cá thể (chuỗi).
        generation_scale (int): Số lượng cá thể trong quần thể.

    Returns:
        list: Danh sách các chuỗi ngẫu nhiên.
    """

    pool = [] 
    for _ in range(generation_scale) : 
        pool.append(''.join(random.sample(char_list, length))) 
    return pool 


def generate_offspring(p1: str, p2: str, char_list: list):
    """
    Hàm này tạo ra hai cá thể con bằng cách lai ghép và đột biến từ hai cá thể cha (p1 và p2).

    Args:
    p1 (str): Cá thể cha thứ nhất (chuỗi ký tự).
    p2 (str): Cá thể mate (chuỗi ký tự).
    char_list (list): Danh sách các ký tự có thể thay thế khi thực hiện đột biến.

    Returns:
    tuple: Hai cá thể con sau khi lai ghép và đột biến, dưới dạng chuỗi ký tự.
    """
     
    assert len(p1) == len(p2), "Hai cá thể phải cùng chiều dài!"
    p1_list = list(p1)
    p2_list = list(p2)

    # Lai ghép
    cross_loc = random.randint(0, len(p1) - 1)
    p1_list[0:cross_loc] = p2[0:cross_loc]
    p2_list[0:cross_loc] = p1[0:cross_loc]

    # Đột biến
    vari_loc = random.randint(0, len(p1) - 1)
    vari_char = random.choice(char_list)
    p1_list[vari_loc] = vari_char
    p2_list[vari_loc] = vari_char

    p1_offspring = ''.join(p1_list)
    p2_offspring = ''.join(p2_list)

    return p1_offspring, p2_offspring


def select(target_sentence, sentence, pool, generation_scale, score_dict, tokenizer, text_encoder, tour_size, mask):  
    pool_score = [] 
    target_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    query_time = 1 
    
    # Compute fitness for pool
    for candidate in pool: 
        if candidate in score_dict.keys(): 
            temp_score = score_dict[candidate] 
            pool_score.append((temp_score,candidate)) 
            continue
        candidate_prompt = sentence + ' ' + candidate
        temp_score = cos_embedding_text(target_embedding, candidate_prompt, tokenizer=tokenizer, text_encoder=text_encoder)
        query_time+=1
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

    return selected_pool, query_time
    

def evolution_strategy(target_sentence, sentence, char_list, length, generation_num, generation_scale, tokenizer, text_encoder, tour_size, mask = None) : 
    generation_list = init_pool(char_list, length, generation_scale)
    total_query_time = 0 
    result = [] 
    score_dict = {} 

    for _ in range(generation_num): 
        tem_pool = generation_list 
        indices = np.arange(len(generation_list)) 
        random.shuffle(indices) 

        # Chọn 2 cha mẹ ngẫu nhiên để lai ghép 
        for i in range(0, len(generation_list), 2):
            candidate = generation_list[indices[i]] 
            mate = generation_list[indices[i+1]] 
            g1, g2 = generate_offspring(candidate, mate, char_list)
            tem_pool.append(g1)
            tem_pool.append(g2)
        
        generation_list, query_time = select(target_sentence = target_sentence, sentence=sentence, pool=tem_pool, generation_scale =generation_scale , score_dict=score_dict, tokenizer=tokenizer, text_encoder=text_encoder, tour_size=tour_size, mask=mask)
        total_query_time += query_time

    res = sorted(score_dict.items(),key = lambda x:x[1],reverse = True)[0:3]
    return res
