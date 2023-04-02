from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, Conv1D
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)
print(y.shape)
x = x.reshape(-1,8,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
# Scaler = MinMaxScaler()
# Scaler.fit(x_train)
# x_train = Scaler.transform(x_train)
# x_test = Scaler.transform(x_test)

print(x_train.shape)


# 2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(8, 1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# hamsu
# loss :  [0.582683801651001, 0.0019376715645194054]
# r2 :  0.5593369929866568