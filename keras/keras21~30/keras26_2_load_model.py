from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))     # <class 'numpy.ndarray'>
print(x)

print(np.min(x), np.max(x))     # 0.0 711.0


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_test), np.max(x_test))     # 0.0 1.0


# 2. 모델

# model = Sequential()
# # model.add(Dense(1, input_dim=13))
# model.add(Dense(30, input_shape=(13,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

# input1 = Input(shape=(13,))
# dense1 = Dense(30)(input1)
# dense2 = Dense(20)(dense1)
# dense3 = Dense(10)(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)

# model.save('./_save/keras26_1_save_model.h5')

model = load_model('./_save/keras26_1_save_model.h5')
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
