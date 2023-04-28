from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, SimpleRNN, Conv1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########### 실습 #############
# scaler=MinMaxScaler()
# x_train = x_train.reshape(-1,1)
# print(x_train.shape)
# print((x_train.shape[0]//(28*28)))
# x_train = scaler.fit_transform(x_train)
# x_test = x_test.reshape(-1,1)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

y_train=np.array(pd.get_dummies(y_train))
y_test=np.array(pd.get_dummies(y_test))


x_train = x_train
# print(np.unique(y_train, return_counts=True))
# print(type(x_train))
print(x_train)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(7, 2, input_shape=(28,28)))  
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='min', patience=100, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=50, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')

# import matplotlib.pyplot as plt 

# plt.plot(hist.history['val_acc'], label='val_acc')
# plt.show()