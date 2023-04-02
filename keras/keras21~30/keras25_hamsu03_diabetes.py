from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=10))
# model.add(Dense(64))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(10,))
dense1 = Dense(32)(input1)
dense2 = Dense(64)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(32)(dense3)
dense5 = Dense(8)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.2, callbacks=[es], verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# hamsu
# loss :  2921.799560546875
# r2 :  0.5086695629869328