import torch 
import numpy as np 

def get_char_table(c = None):
    char_table=['·','~','!','@','#','$','%','^','*','(',')','=','-','*','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('a'),ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    if c != None : 
        char_table = [i for i in char_table if i not in [c]]
    return char_table

def get_text_embeds_without_uncond(prompt, tokenizer, text_encoder):
    # Tokenize text and get embeddings
    text_input = tokenizer(
      prompt, padding='max_length', max_length=tokenizer.model_max_length,
      truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    return text_embeddings

def cos_embedding_text(embading, text, mask=None, tokenizer=None, text_encoder=None):    
    change_embading = get_text_embeds_without_uncond([text], tokenizer, text_encoder)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    if mask==None:
        return cos(embading.view(-1), change_embading.view(-1)).item()
    else:
        return cos(embading.view(-1)*mask, change_embading.view(-1)*mask).item()
    
