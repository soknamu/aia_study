from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)     # (581012, 54), (581012,)
print(type(x), type(y))

print(np.unique(y))     # [1 2 3 4 5 6 7]
y = pd.get_dummies(y)
print(type(y))
y = np.array(y)
print(type(y))
print(y.shape)          # [581012, 7]

# y = to_categorical(y)     # (0이 추가됨)
# print(y.shape)      # (581012, 8) 이 되므로 첫열을 하나 제거하거나, 다른거 쓰거나

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, stratify=y)
print(y_train)
print(np.unique(y_train))

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=54))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2, verbose=1)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print("result : ", result)

y_predict = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)