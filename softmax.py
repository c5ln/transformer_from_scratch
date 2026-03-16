import numpy as np


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)


# axis = -1은 배열의 가장 마지막 차원을 가리킨다. 딥러닝에서 마지막 차원은 
# 항상 데이터의 가장 기본 단위 요소들이 나열된 축이다. 
# 어텐션 행렬에서는 'Query 토큰이 바라보는 모든 Key Token들의 Sequence가 여기에 해당한다.


# Keepdims=True
# 차원 축소 연산(max, sum)은 기본적으로 연산이 수행된 축을 파괴한다.
# 차원이 파괴되면 파이썬의 브로드캐스팅 규칙이 개입하면서 완전히 엉뚱한 방향으로 나눗셈이 발생한다.

