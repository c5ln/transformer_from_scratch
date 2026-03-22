import scaled_dot as sd
import numpy as np

# W[0] = W_Q, W[1] = W_K, W[2] = W_V, X = input, n = head 개수
def multi_head(X, W, W_O, n, mask=None):
    
    scaled_res = []
    for i in range (0,n) :
        Q_projected = X @ W[i][0]
        K_projected = X @ W[i][1]
        V_projected = X @ W[i][2]
        scaled_res.append(sd.scaled_dot_product(Q_projected, K_projected, V_projected, mask))
    
    concat_res = concat(scaled_res, n)
    
    return concat_res @ W_O # linear
    

def concat(scaled_res,n):
    concatenated = []
    concatenated = np.concatenate(scaled_res, axis=-1)
    
    return concatenated