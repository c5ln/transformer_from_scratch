import numpy as np
import softmax as soft

def scaled_dot_product(Q,K,V,mask = None):
    qk = np.dot(Q,K.T)
    scaled_qk = qk / np.sqrt(K.shape[-1])
    if mask is not None:
        scaled_qk += mask # -inf를 더해서 softmax후 0이 되도록
    softmax_qk = soft.softmax(scaled_qk)
    result = np.dot(softmax_qk,V)
    return result 