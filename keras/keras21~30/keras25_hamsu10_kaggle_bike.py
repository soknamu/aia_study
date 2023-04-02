import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


# 1.데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()

# 1.4 x, y 분리
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
# 2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=8, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(32, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(8, activation='relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=100)
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 :', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 4.1 내보내기
y_submit = model.predict(test_csv)
submission= pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit

submission.to_csv(path_save + 'submit_0310_0728.csv')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')
plt.title('케글바이크')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.grid()
plt.legend()
plt.show()

# loss :  22537.240234375
# r2 스코어 : 0.3062697116866663
# RMSE :  150.12405472335942

# (MinMaxScaler) 
# loss :  21739.61328125
# r2 스코어 : 0.33082160498660074
# RMSE :  147.44359560578232

# (StandardScaler) 
# loss :  22303.5390625
# r2 스코어 : 0.3134631591768615
# RMSE :  149.34369123864897

# (MaxAbsSclaer) 
# loss :  22578.4609375
# r2 스코어 : 0.3050008816524633
# RMSE :  150.26128016571516

# (RobustScaler)
# loss :  21702.431640625
# r2 스코어 : 0.33196616970960546
# RMSE :  147.3174476588654