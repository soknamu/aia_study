# 저장할때 평가결과값, 훈련시간을 파일에 넣기

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
input1 = Input(shape=(13,))
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import datetime
date = datetime.datetime.now()
print(date)
# 2023-03-14 11:11:02.290435
date = date.strftime("%m%d_%H%M")
print(date)
# 0314_1115

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs=100, callbacks=(es, mcp), validation_split=0.2)

# 4. 평가, 예측
from sklearn.metrics import r2_score

print("================================ 1. 기본 출력 ================================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)