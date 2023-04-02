from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
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

print(x.shape)
x = x.reshape(178, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = StandardScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(13, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()

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

# acc :  0.8888888888888888

# (MinMaxScaler) 
# acc :  0.9814814814814815

# (StandardScaler) 
# acc :  0.9444444444444444

# (MaxAbsSclaer) 
# acc :  0.9259259259259259

# (RobustScaler)
# acc :  0.9629629629629629
