from math import sqrt
from positional_encoding import positional_encoding
import numpy as np

def create_embedding_matrix(voca_size, d_model):
    return np.random.rand(voca_size, d_model)

def embedding(token_indices,embedding_matrix, d_model):
    seq_len = len(token_indices)
    
    # look-up
    token_vectors = embedding_matrix[token_indices]
    
    # scaled
    output = token_vectors * sqrt(d_model)
    
    # positional encoding
    output = output + positional_encoding(seq_len, d_model))
    
    return output