import torch 
import numpy as np 
import gc
from torch import autocast

def get_text_embeds_without_uncond(prompt, tokenizer, text_encoder, batch_size=1024):
    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = list(prompt)

    all_embeddings = []
    print(f"Getting text embeddings for {len(prompts)} prompts")
    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx:start_idx + batch_size]
        text_input = tokenizer(
            batch_prompts,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            batch_embeddings = text_encoder(text_input.input_ids.cuda())[0]

        # Move to CPU and free GPU memory
        batch_embeddings = batch_embeddings.cpu()
        all_embeddings.append(batch_embeddings)

        del text_input
        del batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()

    if len(all_embeddings) == 1:
        return all_embeddings[0]

    # Concat on CPU
    return torch.cat(all_embeddings, dim=0)

def cos_embedding_text_batch(embading, texts, mask=None, tokenizer=None, text_encoder=None, batch_size=2048):
    # Compute cosine similarity between a single target embedding and a batch of texts
    change_embeddings = get_text_embeds_without_uncond(texts, tokenizer, text_encoder, batch_size=batch_size)
    target_flat = embading.view(1, -1)
    batch_flat = change_embeddings.view(change_embeddings.shape[0], -1)
    if mask is not None:
        mask = mask.view(1, -1).to(batch_flat.device)
        target_flat = target_flat * mask
        batch_flat = batch_flat * mask
    return torch.nn.functional.cosine_similarity(batch_flat, target_flat.expand_as(batch_flat), dim=1).tolist()

def cos_embedding_text(embading, text, mask=None, tokenizer=None, text_encoder=None):    
    # Backward-compatible single-text cosine
    return cos_embedding_text_batch(embading, [text], mask=mask, tokenizer=tokenizer, text_encoder=text_encoder)[0]
    
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