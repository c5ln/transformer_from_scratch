import numpy as np
import softmax as soft
from scaled_dot import scaled_dot_product

print("=" * 50)
print("Test 1: Similar Q and K routes to correct V")
print("=" * 50)
# Q[0] is very similar to K[2], so result[0] should be close to V[2]
Q = np.array([[10.0, 0.0],
              [0.0, 10.0],
              [5.0, 5.0],
              [1.0, 1.0]])

K = np.array([[0.0, 1.0],
              [1.0, 0.0],
              [10.0, 0.0],   # <-- very similar to Q[0]
              [0.0, 0.0]])

V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [7.0, 7.0],    # <-- result[0] should be close to this
              [3.0, 3.0]])

result = scaled_dot_product(Q, K, V)
print(f"result[0] = {result[0]}")
print(f"V[2]      = {V[2]}")
print(f"Close?    → {np.allclose(result[0], V[2], atol=0.5)}")


print("\n" + "=" * 50)
print("Test 2: Each row of attention weights sums to 1")
print("=" * 50)
Q2 = np.random.randn(5, 8)
K2 = np.random.randn(5, 8)
V2 = np.random.randn(5, 8)

# Manually compute attention weights to inspect them
qk = np.dot(Q2, K2.T)
scaled_qk = qk / np.sqrt(K2.shape[-1])
attention_weights = soft.softmax(scaled_qk)

row_sums = np.sum(attention_weights, axis=-1)
print(f"Row sums: {row_sums}")
print(f"All ≈ 1? → {np.allclose(row_sums, 1.0)}")


print("\n" + "=" * 50)
print("Test 3: Uniform similarity → output is mean of V")
print("=" * 50)
# All Q and K are identical → every attention weight should be 1/seq_len
same_vec = np.array([[1.0, 1.0]])
Q3 = np.tile(same_vec, (4, 1))  # 4 identical tokens
K3 = np.tile(same_vec, (4, 1))

V3 = np.array([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0],
               [7.0, 8.0]])

result3 = scaled_dot_product(Q3, K3, V3)
expected_mean = np.mean(V3, axis=0)
print(f"result[0]     = {result3[0]}")
print(f"mean of V     = {expected_mean}")
print(f"All rows same? → {np.allclose(result3[0], result3[1]) and np.allclose(result3[1], result3[2])}")
print(f"Equal to mean? → {np.allclose(result3[0], expected_mean)}")


print("\n" + "=" * 50)
print("Test 4: Scaling prevents softmax saturation")
print("=" * 50)
np.random.seed(42)
d_k_large = 512
Q4 = np.random.randn(4, d_k_large)
K4 = np.random.randn(4, d_k_large)

# Without scaling → softmax saturates (nearly one-hot)
qk_unscaled = np.dot(Q4, K4.T)
weights_unscaled = soft.softmax(qk_unscaled)

# With scaling → softmax is smoother
qk_scaled = qk_unscaled / np.sqrt(d_k_large)
weights_scaled = soft.softmax(qk_scaled)

print("Without scaling (row 0):")
print(f"  weights = {weights_unscaled[0]}")
print(f"  max weight = {weights_unscaled[0].max():.6f}")

print("With scaling (row 0):")
print(f"  weights = {weights_scaled[0]}")
print(f"  max weight = {weights_scaled[0].max():.6f}")

print(f"Unscaled more concentrated? → {weights_unscaled.max() > weights_scaled.max()}")