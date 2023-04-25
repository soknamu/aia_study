import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan,  8, np.nan]]
                    ).transpose()


data.columns = ['x1','x2','x3','x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True
#False = 결측치x True = 결측치o

# x1    1
# x2    2
# x3    0
# x4    3
'''
#1. 결측치 삭제
print("============== 결측치 삭제 ==================")
#print(data['x1'].dropna()) #이렇게 하면 그 열에서만 삭제되어서 그게 그거다
print(data.dropna())
print("============== 결측치 삭제 ==================")
print(data.dropna(axis = 0))       #디폴트가 행위주 삭제
print("============== 결측치 삭제 ==================")
print(data.dropna(axis = 1))       #디폴트가 열위주 삭제
'''

#2-1 특정값 - 평균
# print("============== 결측치 처리 mean() ==================")
# means = data.mean()
# print('평균 :' , means)
# data2 = data.fillna(means)
# print(data2)

# ============== 결측치 처리 mean() ==================
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000

# #2-2 특정값 - 중위
# print("============== 결측치 처리 median() ==================")
# median = data.median()
# print('중위값 : ', median)
# data3 = data.fillna(median)
# print(data3)

# ============== 결측치 처리 median() ==================
# 중위값 :  x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
# dtype: float64
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0
#======================================================

#2-3 특정값 - 0
# print("============== 결측치 처리 median() ==================")
# #median = data.fillna()
# #print('중위값 : ', median)
# data4 = data.fillna(0)
# print(data4)

#2-4 특정값 ffill, bfill
# print("============== 결측치 처리 ffill, bfill ==================")
# data4 = data.fillna(method='ffill') #맨위에 값을 못땡겨옴
# data5 = data.fillna(method='bfill') #맨 밑의 값을 못땡겨옴

# print(data4)
# print(data5)

# ============== 결측치 처리 ffill ==================
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#2-5 특정값 임의값 넣기
# print("============== 결측치 처리 -임의의 값으로 채우기 ==================")
# data6 = data.fillna(value = 7777777) #value 빼도가능
# print(data6)

# ============== 결측치 처리 -임의의 값으로 채우기 ==================
#           x1         x2    x3         x4
# 0        2.0        2.0   2.0  7777777.0
# 1  7777777.0        4.0   4.0        4.0
# 2        6.0  7777777.0   6.0  7777777.0
# 3        8.0        8.0   8.0        8.0
# 4       10.0  7777777.0  10.0  7777777.0

############################# 특정칼럼만 !!!!!!!!!!!######################
means = data['x1'].mean()
data['x1'] = data['x1'].fillna(means)

median= data['x2'].median()
data['x2'] = data['x2'].fillna(median)

data['x4'] = data['x4'].fillna(method='ffill')
data['x4'] = data['x4'].fillna(value=777777)
#data['x4'] = data['x4'].fillna(method='ffill').fillna(value=777777)

print(data)
#########################################

#1. x1컬럼에 평균값을 넣고

#2. x2컬럼에 중위값을 넣고

#3. x4컬럼에 ffill한후/ 제일위에 남은 행에 777777로 채우기

#      x1   x2    x3        x4
# 0   2.0  2.0   2.0  777777.0
# 1   6.5  4.0   4.0       4.0
# 2   6.0  4.0   6.0       4.0
# 3   8.0  8.0   8.0       8.0
# 4  10.0  4.0  10.0       8.0