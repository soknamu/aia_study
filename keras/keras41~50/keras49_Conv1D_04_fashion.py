from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# [실습]

# 1. 데이터
# reshape 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

print(np.unique(y_train))
y_train = y_train.reshape(60000,)
y_test = y_test.reshape(10000,)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = np.array(pd.get_dummies(y_train))
y_test = np.array(pd.get_dummies(y_test))

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(28, 28)))
# model.add(Dense(64, input_shape=(28*28,)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)
