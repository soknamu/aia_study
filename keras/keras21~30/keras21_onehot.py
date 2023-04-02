# [과제]

# 3가지 원핫인코딩 방식을 비교할 것

#1. pandas의 get_dummies


import pandas as pd
y=pd.get_dummies(y)
print(y.shape)

#판다스는 get_dummies 를 통해서 된다.

#2. keras의 to_categorical
 
from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
y = to_categorical(y)
print(y.shape) #(178, 3)

# 텐서플로의 단점은 라벨값이 1부터 시작되면 텐서플로는 0부터 인식하기 때문에 sklearn이랑 pandas와 
#  1차이 난다. 그래서 y = np.delete(y, 0, axis=1) 라벨값을 없애주는 식을 쓴다면 sklearn이랑 판다스처럼
# 라벨값이 같아진다.

#3. sklearn 의 OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y.shape)


# 미세한 차이를 정리하시오!!!
