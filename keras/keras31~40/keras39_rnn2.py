import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# y = ?

x = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7], [5,6,7,8], [6,7,8,9]])

y = np.array([5,6,7,8,9,10])

print(x.shape, y.shape)     # (6, 4) (7,)

# x의 shape = (행, 열, 몇개씩 훈련하는지)
x = x.reshape(6, 4, 1)

# 2. 모델
model = Sequential()
model.add(SimpleRNN(32, input_shape=(4, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x, y, epochs=10000, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([7,8,9,10]).reshape(1, 4, 1)     # [[[7], [8], [9], [10]]]
print(x_predict.shape)

result = model.predict(x_predict)
print('loss : ', loss)
print('[7,8,9,10]의 결과 : ' , result)



