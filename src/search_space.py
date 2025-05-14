def get_char_table(c = None):
    char_table=['·','~','!','@','#','$','%','^','*','(',')','=','-','*','.','<','>','?',',','\'',';',':','|','\\','/']
    for i in range(ord('a'),ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    if c != None : 
        char_table = [i for i in char_table if i not in [c]]
    return char_table