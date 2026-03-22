
from embedding import create_embedding_matrix, embedding
from encoder import encoder_layer
from decoder import decoder_layer
from softmax import softmax
def transformer(token_indices,voca_size,encoding_attention_weights,decoding_attention_weights, encoding_W_O,decoding_W_O,stack_n,decoding_ffn_w,decoding_ffn_b,gamma,beta,W_out,d_model=512,n_heads=8):
    # embedding matrix 생성
    embedding_matrix = create_embedding_matrix(voca_size,d_model)
    # embedding, positional encoding
    token_vectors = embedding(token_indices,embedding_matrix,d_model)
    # encoding
    enc_out = token_vectors
    for i in range (0,stack_n):
        enc_out = encoder_layer(enc_out,encoding_attention_weights[i],encoding_W_O[i],n_heads)
    
    # decoding
    dec_out = token_vectors
    for i in range (0,stack_n):
        dec_out = decoder_layer(dec_out,enc_out,decoding_attention_weights[i][0], decoding_attention_weights[i][1], decoding_W_O[i][0], decoding_W_O[i][1], n_heads,decoding_ffn_w,decoding_ffn_b,gamma,beta)

    output = dec_out @ W_out
    result = softmax(output)
    return result