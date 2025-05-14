import torch 
import numpy as np 
import gc
from torch import autocast

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
    
def compare_sentences(sentence1,sentence2,mask=None,tokenizer=None,text_encoder=None) : 
    text_embedding1 = get_text_embeds_without_uncond([sentence1], tokenizer, text_encoder)
    text_embedding2 = get_text_embeds_without_uncond([sentence2], tokenizer, text_encoder)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    if mask != None : 
        result = cos(text_embedding1.view(-1) *mask, text_embedding2.view(-1)*mask).item()
    else : 
        result = cos(text_embedding1.view(-1) , text_embedding2.view(-1)).item()
    return result

def get_char_table():
    char_table=['·','~','!','@','#','$','%','^','*','(',')','=','-','_','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('a'),ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    return char_table
    
def generate_images(prompts,pipe,generator,num_image = 10) : 
    torch.cuda.empty_cache()
    gc.collect()
    images = [] 
    for prompt in prompts : 
        with autocast('cuda') : 
            image = pipe([prompt],generator = generator,num_inference_steps=50,num_images_per_prompt = num_image).images
            images.append(image)
    return images