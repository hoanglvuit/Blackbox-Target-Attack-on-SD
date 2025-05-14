import numpy as np 
import torch
from .utils import get_text_embeds_without_uncond

def consine_similarity(embed1, embed2, mask=None): 
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6) 
    embed1_flat = embed1.view(-1) 
    embed2_flat = embed2.view(-1)
    if mask is not None: 
        return cos(embed1_flat * mask, embed2_flat * mask).item() 
    return cos(embed1_flat, embed2_flat).item()

def find_target_embedding_mask(target_embedding, ori_embedding, thres=1): 
    diff = torch.abs(target_embedding - ori_embedding) 
    mask = diff > thres 
    return mask.float() 

def auto_tune_threshold(target_sentence,sentence,tokenizer,text_encoder,start = 1,end = 10, space_limit = 2000)  : 
    target_embedding = get_text_embeds_without_uncond(target_sentence, tokenizer=tokenizer, text_encoder=text_encoder)
    ori_embedding = get_text_embeds_without_uncond(sentence, tokenizer=tokenizer, text_encoder=text_encoder)
    thres = np.arange(start,end,0.01) 
    for thre in thres : 
        te = find_target_embedding_mask(target_embedding,ori_embedding,thres=thre)
        te_list = te.tolist()
        te = te.view(-1) 
        cosine = consine_similarity(target_embedding,ori_embedding,te) 
        if np.sum(te_list) < space_limit or cosine < 0.3 : 
            return te, cosine