import numpy as np
import FFN as ffn

np.random.seed(42)

print("=" * 50)
print("Test 1: Output shape is correct")
print("=" * 50)
seq_len = 4
d_model = 8
d_ff = 16

W = [np.random.randn(d_model, d_ff),   # W[0]: (d_model, d_ff)
     np.random.randn(d_ff, d_model)]    # W[1]: (d_ff, d_model)

b = [np.random.randn(d_ff),             # b[0]: (d_ff,)
     np.random.randn(d_model)]          # b[1]: (d_model,)

x = np.random.randn(seq_len, d_model)
result = ffn.feed_forward_network(x, W, b)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {result.shape}")
print(f"Expected:     ({seq_len}, {d_model})")
print(f"Correct?    → {result.shape == (seq_len, d_model)}")


print("\n" + "=" * 50)
print("Test 2: ReLU zeroes out negative values")
print("=" * 50)
# After first layer, check that no negative values survive ReLU
hidden = x @ W[0] + b[0]
after_relu = np.maximum(0, hidden)

num_negative_before = np.sum(hidden < 0)
num_negative_after = np.sum(after_relu < 0)

print(f"Negatives before ReLU: {num_negative_before}")
print(f"Negatives after ReLU:  {num_negative_after}")
print(f"All non-negative?    → {num_negative_after == 0}")


print("\n" + "=" * 50)
print("Test 3: Manual computation matches function output")
print("=" * 50)
# Manually compute: max(0, x @ W[0] + b[0]) @ W[1] + b[1]
manual_hidden = x @ W[0] + b[0]
manual_relu = np.maximum(0, manual_hidden)
manual_result = manual_relu @ W[1] + b[1]

print(f"Function result[0]: {result[0][:4]}")
print(f"Manual result[0]:   {manual_result[0][:4]}")
print(f"Equal?            → {np.allclose(result, manual_result)}")


print("\n" + "=" * 50)
print("Test 4: Each token is processed independently")
print("=" * 50)
# Running token 0 alone should give the same result as running all tokens
single_token = x[0:1]  # shape (1, d_model)
single_result = ffn.feed_forward_network(single_token, W, b)

print(f"Full batch result[0]:   {result[0][:4]}")
print(f"Single token result[0]: {single_result[0][:4]}")
print(f"Equal?                → {np.allclose(result[0], single_result[0])}")


print("\n" + "=" * 50)
print("Test 5: Zero input produces bias-only output")
print("=" * 50)
# If x is all zeros: hidden = b[0], then ReLU, then @ W[1] + b[1]
x_zero = np.zeros((1, d_model))
zero_result = ffn.feed_forward_network(x_zero, W, b)
expected = np.maximum(0, b[0]) @ W[1] + b[1]

print(f"Zero input result: {zero_result[0][:4]}")
print(f"Expected (bias):   {expected[:4]}")
print(f"Equal?           → {np.allclose(zero_result[0], expected)}")