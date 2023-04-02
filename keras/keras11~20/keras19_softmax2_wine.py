from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# OneHotEncoder를 사용하여 y를 원핫인코딩합니다.
encoder = OneHotEncoder()
y_2d = y.reshape(-1, 1)
print(y_2d.shape)
y = encoder.fit_transform(y_2d).toarray()
print(y.shape)
# y_2d = y.reshape(-1, 1)
# y = OneHotEncoder().fit_transform(y_2d).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)


# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict.shape)

y_true = np.argmax(y_test, axis=-1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_predict)
print('acc : ', acc)