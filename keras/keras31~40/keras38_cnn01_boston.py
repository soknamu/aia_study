from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score
# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))     # <class 'numpy.ndarray'>
print(x)

print(np.min(x), np.max(x))     # 0.0 711.0

print(x.shape)

x = x.reshape(506, 13, 1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)
# scaler = MinMaxScaler()

# scaler = StandardScaler()

# scaler = MaxAbsScaler()

print(x_train.shape)

# x_train = x_train.reshape(-1,13)
# x_test = x_test.reshape(-1,13)
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(-1, 13, 1, 1)
# x_test = x_test.reshape(-1, 13, 1, 1)
print(np.min(x_test), np.max(x_test))     # 0.0 1.0


# 2. 모델
model = Sequential()
model.add(Conv2D(64, 2, padding='same', input_shape=(13, 1, 1)))
model.add(Conv2D(10, 2, padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)