from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
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
model.add(Dense(64, input_dim=64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = (model.predict(x_test))
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict.shape)

y_true = np.argmax(y_test, axis=-1)

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