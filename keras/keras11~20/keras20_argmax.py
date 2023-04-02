import numpy as np

a = np.array([[1,2,3,], [6,4,5], [7,9,2], [3,2,1], [2,3,1]])

print(a)
print(a.shape)      # (5, 3)
print(np.argmax(a))     # 7
print(np.argmax(a, axis=0))     # [ 2 2 1 ]  axis=0  행끼리 비교함
print(np.argmax(a, axis=1))     # [ 2 0 1 0 1 ]  axis=1  그래서 열끼리 비교함
print(np.argmax(a, axis=-1))    # [ 2 0 1 0 1 ]  -1은 가장 마지막이라는 뜻
    # 가장 마지막 축, 이건 2차원이니까 가장 마지막 축은 1
    # 그래서 -1을 쓰면 이 데이터는 1과 동일 


