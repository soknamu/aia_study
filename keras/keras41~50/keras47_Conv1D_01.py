import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten

# 1. 데이터

# 2. 모델
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))            # Total Param : 541
model.add(Conv1D(10, 2, input_shape=(3,1)))         # Total Param : 141
model.add(Conv1D(10, 2, input_shape=(3,1)))         # Total Param : 301
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary()