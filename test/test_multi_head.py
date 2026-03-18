import numpy as np
import softmax as soft
import scaled_dot as sd
import multi_head as mh

np.random.seed(42)

print("=" * 50)
print("Test 1: Output shape is correct")
print("=" * 50)
# 4 tokens, d_model=8, 2 heads → d_k=d_v=4
seq_len = 4
d_model = 8
n_heads = 2
d_k = d_model // n_heads  # 4

# W[i][0]=W_Q, W[i][1]=W_K, W[i][2]=W_V, each (d_model, d_k)
W = []
for i in range(n_heads):
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)
    W.append([W_Q, W_K, W_V])

W_O = np.random.randn(d_model, d_model)  # (n_heads * d_v, d_model)
X = np.random.randn(seq_len, d_model)

result = mh.multi_head(X, W, W_O, n_heads)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {result.shape}")
print(f"Expected:     ({seq_len}, {d_model})")
print(f"Correct?    → {result.shape == (seq_len, d_model)}")


print("\n" + "=" * 50)
print("Test 2: Single head = scaled dot product + W_O")
print("=" * 50)
# With 1 head, multi_head should equal: (X @ W_Q, X @ W_K, X @ W_V) → attention → @ W_O
n_heads_1 = 1
d_k_1 = d_model  # single head uses full dimension

W_single = []
W_Q1 = np.random.randn(d_model, d_k_1)
W_K1 = np.random.randn(d_model, d_k_1)
W_V1 = np.random.randn(d_model, d_k_1)
W_single.append([W_Q1, W_K1, W_V1])

W_O1 = np.random.randn(d_k_1, d_model)

# Multi-head result
mh_result = mh.multi_head(X, W_single, W_O1, n_heads_1)

# Manual result
Q_manual = X @ W_Q1
K_manual = X @ W_K1
V_manual = X @ W_V1
manual_result = sd.scaled_dot_product(Q_manual, K_manual, V_manual) @ W_O1

print(f"Multi-head result[0]: {mh_result[0][:4]}")
print(f"Manual result[0]:     {manual_result[0][:4]}")
print(f"Equal?              → {np.allclose(mh_result, manual_result)}")


print("\n" + "=" * 50)
print("Test 3: Different heads produce different outputs")
print("=" * 50)
# Each head has different weights, so their intermediate results should differ
Q_h0 = X @ W[0][0]
K_h0 = X @ W[0][1]
V_h0 = X @ W[0][2]
head0_out = sd.scaled_dot_product(Q_h0, K_h0, V_h0)

Q_h1 = X @ W[1][0]
K_h1 = X @ W[1][1]
V_h1 = X @ W[1][2]
head1_out = sd.scaled_dot_product(Q_h1, K_h1, V_h1)

print(f"Head 0 output[0]: {head0_out[0]}")
print(f"Head 1 output[0]: {head1_out[0]}")
print(f"Different?      → {not np.allclose(head0_out, head1_out)}")


print("\n" + "=" * 50)
print("Test 4: W_O actually affects the output")
print("=" * 50)
# Same input and head weights, different W_O → different final output
W_O_a = np.random.randn(d_model, d_model)
W_O_b = np.random.randn(d_model, d_model)

result_a = mh.multi_head(X, W, W_O_a, n_heads)
result_b = mh.multi_head(X, W, W_O_b, n_heads)

print(f"Result with W_O_a[0]: {result_a[0][:4]}")
print(f"Result with W_O_b[0]: {result_b[0][:4]}")
print(f"Different?          → {not np.allclose(result_a, result_b)}")