import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])
# print(x)
# print(y)
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=2)

# 4. 평가, 예측
lose = model.evaluate(x_test, y_test)
print("lose : ", lose)

result = model.predict([11])
print("[11]의 예측값 : ", result)

"""
lose :  3.78956116703702e-13
1/1 [==============================] - 0s 74ms/step
[11]의 예측값 :  [[11.]]

([10,100,1000,1000,10,1], mse, adam, 1000, 2)
"""