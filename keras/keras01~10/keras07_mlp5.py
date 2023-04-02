# x의 열은 3개
# y의 열은 2개

import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)])    # (3, 10)
x = x.T   # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]) # (2, 10)

y = y.T     #(10, 2)


# [실습]
# 예측 : [[9, 30, 210]] 


#2.  모델구성
model = Sequential()
model.add(Dense(20, input_dim=3))
model.add(Dense(400))
model.add(Dense(2000))
model.add(Dense(400))
model.add(Dense(20))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict([[9, 30, 210]])
print("[[9, 30 , 210]]의 예측값 : ", result)


"""
loss :  1.4063417314372217e-12
[[9, 30 , 210]]의 예측값 :  [[10.000002  1.8999996]]

([10,100,1000,100,10,2], mse, adam, 1000, 1)
"""

"""
loss :  1.3516299624677375e-12
[[9, 30 , 210]]의 예측값 :  [[10.         1.9000006]]

([20,400,2000,400,20,2], mse, adam, 1000, 1)
"""