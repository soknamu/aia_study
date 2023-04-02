from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

print(x_train.shape)
# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(32)(input1)
dense2 = Dense(64)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(32)(dense3)
dense5 = Dense(8)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# hamsu
# loss :  [0.582683801651001, 0.0019376715645194054]
# r2 :  0.5593369929866568