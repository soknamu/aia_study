from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=20580)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2)

print(hist.history)

plt.plot(hist.history['loss'])
plt.show()