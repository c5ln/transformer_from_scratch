import numpy as np
def feed_forward_network(x,W,b):
    return np.maximum(0, x @ W[0] + b[0]) @ W[1] + b[1]

# W[0] shape : (d_model, d_ff), b[0] shape : (d_ff)





