import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
data = load_breast_cancer()
x = data.data
y = data.target

# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=30))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1,activation = 'sigmoid'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics =['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='acc', patience=20, mode='max', verbose=1)
rlr = ReduceLROnPlateau(monitor='acc', patience=10, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2,
          callbacks=[es, rlr])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("lr:", learning_rate, "loss:", results)
# lr: 0.005 loss: [0.09837311506271362, 0.9736841917037964]
