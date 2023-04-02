from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

# 1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])

# 실습 :: 자르기
x_val = x_train[13:]
y_val = y_train[13:]
x_test = x_train[10:13]
y_test = y_train[10:13]
print(x_val, y_val, x_test, y_test)

# 2. 모델
model = Sequential()
model.add(Dense(32,activation='linear', input_dim=1))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

# # 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)