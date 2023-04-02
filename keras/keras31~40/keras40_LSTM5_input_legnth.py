import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y = ?

x = np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7], [4,5,6,7,8], [5,6,7,8,9]])

y = np.array([6,7,8,9,10])

print(x.shape, y.shape)     # (5, 5) (5,)

# x의 shape = (행, 열, 몇개씩 훈련하는지)
x = x.reshape(5, 5, 1)

# 2. 모델
model = Sequential()                    # [batch, timesteps, feature]
                                        # [batch, input_length, input_dim]
# model.add(LSTM(10, input_shape=(5, 1)))

model.add(LSTM(10, input_length=5, input_dim=1))
# model.add(LSTM(10, input_dim=1, input_length=5))        # 가독성 떨어짐

# 4 * {(10*10) + (1*10) + 10} = 480

# (10*10) + (1*10) + 10 = 120 : RNN
# units * (feature + bias + units ) = params


# model.add(SimpleRNN(15, input_shape=(5, 1)))

# (15*15) + (1*15) + 15 = 255

model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=10000, restore_best_weights=True)
model.fit(x, y, epochs=10, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([6,7,8,9,10]).reshape(1, 5, 1)     # [[[7], [8], [9], [10]]]
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[6,7,8,9,10]의 결과 : ' , result)



