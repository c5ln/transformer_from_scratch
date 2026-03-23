import scaled_dot as sd
import numpy as np

# Q: decoder 출력, K/V: encoder 출력

def cross_attention(Q,K,V,W,W_O,n):
    scaled_res = []
    for i in range (0,n) :
        
        Q_proj = Q @ W[i][0]
        K_proj = K @ W[i][1]
        V_proj = V @ W[i][2]
        
        scaled_res.append(sd.scaled_dot_product(Q_proj, K_proj, V_proj))
    
    concat_res = concat(scaled_res)
    return concat_res @ W_O
    

def concat(scaled_res):
    concatenated = np.concatenate(scaled_res, axis=-1)
    return concatenated