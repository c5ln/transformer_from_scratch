import numpy as np

def layer_norm(x, gamma, beta, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta
    