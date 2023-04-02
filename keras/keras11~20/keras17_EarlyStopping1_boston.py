from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)     #(506, 13), (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.7)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=1000, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es])

"""print("=========================================")
print(hist)
print("=========================================")
# <tensorflow.python.keras.callbacks.History object at 0x000001DD1A4EBB50>
print(hist.history)
print("=========================================")
print(hist.history['loss'])
"""
print("================== 발로스 ====================")
print(hist.history['val_loss'])
print("================== 발로스 ====================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

import matplotlib.pylab as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.grid()
plt.legend()
plt.show()