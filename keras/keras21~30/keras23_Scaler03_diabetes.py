from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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
model = Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

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
"""
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('diabetes')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.grid()
plt.legend()
plt.show()
"""

# loss :  2952.0224609375
# r2 :  0.5035871775477561

# (MinMaxScaler) 
# loss :  2929.993896484375
# r2 :  0.507291603258532

# (StandardScaler) 
# loss :  3040.89501953125
# r2 :  0.48864241810660947

# (MaxAbsSclaer) 
# loss :  2982.55712890625
# r2 :  0.49845250812221

# (RobustScaler)
# loss :  2963.3427734375
# r2 :  0.501683603261612
