
# tokens -> embedding -> positional encoding -> encoder_layer 

from multi_head import multi_head
from norm import layer_norm
from FFN import feed_forward_network
def encoder_layer(x, attention_weights, W_O, n_heads, ffn_W, ffn_b, gamma1, beta1, gamma2, beta2):

    # attention    
    att_out = multi_head(x,attention_weights,W_O,n_heads)

    # add + norm
    x2 = layer_norm(x+att_out, gamma1, beta1)
    
    # ffn
    ffn_out = feed_forward_network(x2, ffn_W,ffn_b)
    
    # add + norm
    x3 = layer_norm(x+ffn_out,gamma2,beta2)
    
    return x3