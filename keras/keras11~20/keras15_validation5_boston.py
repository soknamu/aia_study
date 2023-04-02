import numpy as np
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13), (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=29)

# 2. 모델
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)