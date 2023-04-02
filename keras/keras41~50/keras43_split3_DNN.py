import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

dataset = np.array(range(1, 101))
timesteps = 5
x_predict = np.array(range(96, 106))

# 100 ~ 106 예상값
# 이런 시계열 데이터를 만들기
# 96, 97, 98, 99
# 97, 98, 99, 100
# 98, 99, 100, 101
# ...
# 102, 103, 104, 105


def split_x(dataset, timesteps):
    gen=(dataset[i : (i + timesteps)] for i in range(len(dataset) - timesteps + 1))
    return np.array(list(gen))

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)
x = bbb[:, :-1]
y = bbb[:, 4:5]     # (6, 1)

print(x.shape)      # (96, 4)
print(y.shape)      # (96, 1)

a = split_x(x_predict, timesteps)
print(a)

a1 = a[: , :4]
print(a1)

x = x.reshape(-1, 4)

a1 = a1.reshape(-1, 4)

# 2. 모델
model = Sequential()
model.add(Dense(64, input_shape=(4,)))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.fit(x, y, epochs=100, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)

result = model.predict(a1)
print('loss : ', loss)
print('result : ', result)

