from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Conv2D, Flatten, SimpleRNN, Conv1D, MaxPooling2D, Reshape
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


x_train = x_train.reshape(60000, 28, 28, 1)/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

# print(np.unique(y_train, return_counts=True))
# print(type(x_train))
print(x_train)

# 2. 모델구성
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3)))
# model.add(Conv2D(10, 3))
# model.add(MaxPooling2D())
# model.add(Flatten())        # (N, 250)
# model.add(Reshape(target_shape=(25, 10)))
# model.add(Conv1D(10, 3, padding='same'))
# model.add(LSTM(784))
# model.add(Reshape(target_shape=(28, 28, 1)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(28, 28, 1))
dense1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input1)
dense2 = MaxPooling2D()(dense1)
dense3 = Conv2D(32, 3)(dense2)
dense4 = Conv2D(10, 3)(dense3)
dense5 = MaxPooling2D()(dense4)
dense6 = Flatten()(dense5)
dense7 = Reshape(target_shape=(25, 10))(dense6)
dense8 = Conv1D(10, 3, padding='same')(dense7)
dense9 = LSTM(784)(dense8)
dense10 = Reshape(target_shape=(28, 28, 1))(dense9)
dense11 = Conv2D(32, 3)(dense10)
dense12 = Flatten()(dense11)
output1 = Dense(10, activation='softmax')(dense12)
model = Model(inputs=input1, outputs=output1)

model.summary()


