import numpy as np

a = np.array([[1,2,3], [6,4,5], [7,9,2], [3,2,1],[2,3,1]])
print(a) #5행 3열
print(a.shape) #(5, 3)
print(np.argmax(a)) #7번째 자리가 가장크다.(가장높은 자리의 위치가 나옴(7번째 위치))
print(np.argmax(a, axis =0)) #0은 행이야, 그래서 행끼리 비교 [2,2,1]
#[ [1 2 3]  
#  [6 4 5]
#  [7 9 2]
#  [3 2 1]
#  [2 3 1]]
#
print(np.argmax(a, axis =1)) #[2 0 1 0 1] 1은 열, 그래서 열끼리 비교.
print(np.argmax(a, axis =-1)) #[2 0 1 0 1] -1은 가장 마지막이란 뜻,

#가장마지막 축, 2차원이기 때문에 가장 마지막축은  1
#그래서 -1을 쓰면 이 데이터는 1과 동일
# 3,4 차원에서는 아그맥스를 잘 안씀.