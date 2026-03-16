import numpy as np
import softmax as soft

def scaled_dot_product(Q,K,V):
    qk = np.dot(Q,K.T)
    scaled_qk = qk / np.sqrt(K.shape[-1])
    softmax_qk = soft.softmax(scaled_qk)
    result = np.dot(softmax_qk,V)
    return result 