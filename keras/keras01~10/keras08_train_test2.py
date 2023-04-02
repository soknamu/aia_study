import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

# [실습] 넘파이 리스트 슬라이싱 7:3으로 자르기
x_train = x[:7]
# x_train = x[0:7]
# [ 1 2 3 4 5 6 7 ]

x_test = x[7:]
# x_test = x[7:10]
# [ 8 9 10 ]

y_train = y[:7]   
# y_train = y[0:7]   
# [ 10 9 8 7 6 5 4 ]

y_test = y[7:]
# y_test = y[7:10]
# [ 3 2 1 ]

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

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

