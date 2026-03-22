# [Decoder Layer] × N        ← 구현 필요
#   ├─ Masked Self-Attention  (causal mask)
#   ├─ Add & Norm
#   ├─ Cross-Attention        (Q: decoder, K/V: encoder output)
#   ├─ Add & Norm
#   ├─ Feed Forward Network
#   └─ Add & Norm


# tokens -> embedding -> positional encoding -> encoder_layer 

import numpy as np
from multi_head import multi_head
from norm import layer_norm
from FFN import feed_forward_network
from cross_attention import cross_attention

def decoder_layer(x, encoder_out, attention_w1,attention_w2, W_O1, W_O2, n_heads, ffn_W, ffn_b, gamma1, beta1, gamma2, beta2, gamma3, beta3):

    # masked self-attention    
    seq_len = x.shape[0]
    mask = np.triu(np.full((seq_len,seq_len), -np.inf), k=1)
    # np.full -> 전체를 -inf로 채운 행렬 생성
    # np.triu -> 상삼각 부분만 남김
    # k = 1 -> 대각선 위쪽만 -inf로 마스킹
    
    att_out = multi_head(x,attention_w1,W_O1,n_heads,mask)

    # add + norm
    x2 = layer_norm(x+att_out, gamma1, beta1)
    
    # cross-attention
    cross_att_out = cross_attention(x2,encoder_out,encoder_out,attention_w2,W_O2,n_heads)
        
    # add + norm
    x3 = layer_norm(x2+cross_att_out, gamma2, beta2)
    
    # ffn
    ffn_out = feed_forward_network(x3, ffn_W,ffn_b)
    
    # add + norm
    x4 = layer_norm(x3+ffn_out,gamma3,beta3)
    
    return x4