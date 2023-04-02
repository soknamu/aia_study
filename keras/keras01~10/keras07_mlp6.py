import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)])    # (3, 10)
x = x.T   # (10, 3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10)

y = y.T     #(10, 3)

# [실습]
# 예측 : [[9, 30, 210]] 

#2.  모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict([[9, 30, 210]])
print("[[9, 30 , 210]]의 예측값 : ", result)


"""
loss :  1.7157167613718327e-12
[[9, 30 , 210]]의 예측값 :  [[9.9999990e+00 1.8999975e+00 3.0174851e-07]]
([10,100,1000,100,10,3], mse, adam, 1000, 1)
"""