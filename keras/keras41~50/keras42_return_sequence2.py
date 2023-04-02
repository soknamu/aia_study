import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, GRU, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
             [5,6,7], [6,7,8], [7,8,9], [8,9,10],
             [9,10,11], [10,11,12],
             [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])

print(x.shape, y.shape)     # (13, 3) (13,)

# x의 shape = (행, 열, 몇개씩 훈련하는지)
x = x.reshape(13, 3, 1)

# 2. 모델
model = Sequential()
model.add(LSTM(128, input_shape=(3, 1), return_sequences=True))
model.add(GRU(128, return_sequences=True))
model.add(SimpleRNN(128))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x, y, epochs=1500, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = x_predict.reshape(1, 3, 1)
# print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[50,60,70]의 결과 : ' , result)

# [50,60,70]의 결과 :  [[76.7667]]