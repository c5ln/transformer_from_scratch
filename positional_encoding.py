import math as mt
import numpy as np

def positional_encoding(seq_len,d_model=512):
    
    result = []
    for l in range(seq_len):
        row = []
        for i in range(d_model):
            exponent = (i // 2) * 2 / d_model
            if(i%2==0):
                row.append(mt.sin(l/(10000 ** exponent)))
            else:
                row.append(mt.cos(l/(10000 ** exponent)))
        result.append(row)
        
    return np.array(result)
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))