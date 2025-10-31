from .utils import get_text_embeds_without_uncond, cos_embedding_text_batch

def select(pool_score, beam_width): 
    sorted_pool_score = sorted(pool_score, reverse=True) 
    res = [elt[1] for elt in sorted_pool_score[:beam_width]]
    return res 
    
def beam_search(target_sentence, sentence, char_list, length, mask, tokenizer, text_encoder, beam_widths): 
    score_dict = {} 
    target_embedding = get_text_embeds_without_uncond([target_sentence], tokenizer, text_encoder)
    candidates = []
    iter = 0 
    pool_score_log = []

    while iter < length: 
        pool_score = []
        if not candidates: 
            candidates = char_list
        else: 
            candidates = [x + y for x in candidates for y in char_list]
        # Prepare batch for candidates without cached score
        uncached = [c for c in candidates if c not in score_dict]
        if uncached:
            adv_prompts = [sentence + ' ' + c for c in uncached]
            scores = cos_embedding_text_batch(target_embedding, adv_prompts, tokenizer=tokenizer, text_encoder=text_encoder, mask=mask)
            for c, s in zip(uncached, scores):
                score_dict[c] = s
        for candidate in candidates:
            temp_score = score_dict[candidate]
            pool_score.append((temp_score, candidate))
        pool_score_log.append(pool_score)
        candidates = select(pool_score, beam_widths[iter])
        iter += 1
    return score_dict, pool_score_log