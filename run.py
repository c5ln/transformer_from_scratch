from transformer import transformer
import numpy as np

# Hyper-parameters
d_model = 16
n_heads = 2
d_k = 8
d_ff = 32
stack_n = 2
voca_size = 100

# Encoder Weights
## [stack_n][n_heads][W_Q,W_K,W_V]
encoding_attention_weights = [
    [ 
        [
            np.random.randn(d_model, d_k) for _ in range(3) 
        ] 
        for _ in range(n_heads)
    ]
    for _ in range(stack_n)
]
## linear layer
encoding_W_O = [ np.random.randn(d_model, d_model) for _ in range(stack_n)]

# Decoder Weights
## [stack_n][self or cross attention][n_heads][W_Q, W_K, W_V]
decoding_attention_weights = [
    [
        [
            [
                np.random.randn(d_model, d_k) for _ in range(3)
            ] 
            for _ in range (n_heads) 
        ]
        for _ in range(2)
    ]
    for _ in range (stack_n)
]

## linear 
decoding_W_O = [
    [
        np.random.randn(d_model,d_model),
        np.random.randn(d_model,d_model)
    ]
    for _ in range(stack_n)
]


# Decoding FFN Weights ffn_w[0][] = encoder ffn_w, ffn_w[1][] = encoder ffn_w
ffn_w = [
    [
        [ 
            np.random.randn(d_model, d_ff),
            np.random.randn(d_ff,d_model)
        ] 
        for _ in range (stack_n)
    ] for _ in range(2)
]

ffn_b = [ 
    [
        [
            np.zeros(d_ff),
            np.zeros(d_model)
        ]
        for _ in range (stack_n)
    ] for _ in range(2)
]

#LayerNorm [0], [1] 은 encoder layer 전용. 나머지는 decoder
gamma = [ np.ones(d_model), np.ones(d_model), np.ones(d_model), np.ones(d_model), np.ones(d_model)]
beta = [ np.zeros(d_model), np.zeros(d_model), np.zeros(d_model), np.zeros(d_model), np.zeros(d_model)]

# last linear
W_out = np.random.randn(d_model, voca_size)

src = [0, 1, 2]

output = transformer(src, voca_size,
                     encoding_attention_weights,
                     decoding_attention_weights,
                     encoding_W_O,decoding_W_O,
                     stack_n,ffn_w,ffn_b,gamma,beta,W_out,
                     d_model,n_heads)

next_token = np.argmax(output[-1])
print(next_token)