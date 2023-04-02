import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

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
lose :  3.07901843127692e-13
[11]의 예측값 :  [[10.999999]]

([10,100,1000,100,10,1], mse, adam, 1000, 2)
"""