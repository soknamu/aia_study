from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=20, batch_size=20, validation_split=0.2, verbose=1)

print(hist.history)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('')
plt.show()