import numpy as np

aaa = np.array([1,2,3])

bbb = aaa #bbb 는 aaa를 카피함.

bbb[0] = 4
print(bbb) #[4 2 3]
print(aaa) #[4 2 3]

print("============================")
ccc = aaa.copy()
ccc[1] = 7

print(ccc) #[4 7 3]
print(aaa) #[4 2 3]
#aaa의 주소값이 같이 바뀌어버림. 
# 주소값이 공유가되어버림. 그래서 나온 해결책이 .copy()