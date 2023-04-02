from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train/255.
x_test = x_test/255.
print(np.unique(y_train))
# Scaler = MinMaxScaler()
# x_train = x_train.reshape(-1,1)
# x_test = x_test.reshape(-1,1)
# x_train = Scaler.fit_transform(x_train)
# x_test = Scaler.transform(x_test)
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델구성
model = Sequential()
model.add(LSTM(8, input_shape=(96, 32)))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, restore_best_weights=True, patience=100)

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=5000, verbose=1, callbacks=[es], validation_split=0.2)

end_time = time.time()
print("training time : ", round(end_time - start_time, 2))

# 4. 평가, 예측

result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1))
print('acc : ', acc)

import matplotlib.pyplot as plt 

plt.plot(hist.history['val_acc'], label='val_acc')
plt.show()