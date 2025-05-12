from utils import get_text_embeds_without_uncond, cos_embedding_text


def select(pool_score, beam_width): 
    sorted_pool_score = sorted(pool_score, reverse=True) 
    res = [elt[1] for elt in sorted_pool_score[:beam_width]]
    return res 
    
def beam_search(target_sentence, sentence, char_list, length, mask, tokenizer, text_encoder, beam_widths): 
    score_dict = {} 
    target_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    candidates = char_list
    iter = 0 
    pool_score_log = []

    while iter < length : 
        pool_score = []
        for candidate in candidates :  
            if candidate in score_dict.keys(): 
                temp_score = score_dict[candidate] 
                pool_score.append((temp_score,candidate))
                continue
            adv_prompt = sentence + ' ' + candidate
            temp_score = cos_embedding_text(target_embedding, adv_prompt, tokenizer=tokenizer, text_encoder=text_encoder,mask=mask)
            pool_score.append((temp_score, candidate)) 
            score_dict[candidate] = temp_score
        pool_score_log.append(pool_score)
        candidates = select(pool_score, beam_widths[iter])
        iter += 1
    return score_dict, pool_score_log