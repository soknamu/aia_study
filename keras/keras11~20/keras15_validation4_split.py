from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# 실습 :: 자르기
# train_test_split 로만 잘라라
# 10:3:3
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=123, shuffle=True)


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
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)