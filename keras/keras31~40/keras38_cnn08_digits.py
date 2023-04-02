from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (1797, 64), (1797,)

# y = to_categorical(y)
# print(y.shape)     # (1797, 64), (1797,)
import pandas as pd

y = pd.get_dummies(y)
print(y)
print(y.shape)
print(type(y))
y = np.array(y)

x = x.reshape(-1, 8, 8, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
"""
scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
"""
# print(y_train)
print(np.unique(y_train, return_counts=True))

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, 3, padding='same', input_shape=(8, 8, 1)))
model.add(Conv2D(10, 2, padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = (model.predict(x_test))
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict.shape)

y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_predict)
print('acc : ', acc)


# acc :  0.9462962962962963

# (MinMaxScaler) 
# acc :  0.9685185185185186

# (StandardScaler) 
# acc :  0.9666666666666667

# (MaxAbsSclaer) 
# acc :  0.9722222222222222

# (RobustScaler)
# acc :  0.9388888888888889