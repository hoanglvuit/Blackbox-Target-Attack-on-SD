from .utils import compare_sentences


def avoidance_strategy(char_list, tar_object, tokenizer, text_encoder):
    char_collection = list(set(tar_object)) 
    
    min_score = float('inf') 
    min_word = ''
    
    for char in char_collection:
        s_word = tar_object.replace(char, '')  # Remove the character
        score = compare_sentences(tar_object, s_word, None, tokenizer, text_encoder)
        if score < min_score:
            min_word = char
            min_score = score
    
    print('word:', tar_object, 'min_word:', min_word, 'min_score:', min_score)
    
    new_char_list = [c for c in char_list if c != min_word]
    
    return new_char_list
