import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets)
print(len(datasets))
print(datasets.shape)
print(datasets.columns)
print(datasets.info())
print(datasets.describe())
print(type(datasets))

print(datasets['T (degC)'].values)      # 판다스를 넘파이로
print(datasets['T (degC)'].to_numpy)    # 판다스를 넘파이로

# import matplotlib.pyplot as plt
# plt.plot(datasets['T (degC)'].values)
# plt.show()

x = datasets.drop(['T (degC)'], axis=1)
y = datasets['T (degC)']
x = np.array(x)
y = np.array(y)
# y = y.reshape(420551, 1)
# print(x.shape)
# print(y.shape)
datasets = np.column_stack((x,y))
# print(datasets)
# print(datasets.shape)

def split_xy(datasets, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(datasets)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column-1
        if y_end_number > len(datasets):
            break
        tmp_x = datasets[i:x_end_number, :-1]
        tmp_y = datasets[x_end_number:y_end_number+1, -1]
        x.append(tmp_x)
        y.append(tmp_y)
        
    return np.array(x), np.array(y)

# def split_xy(datasets, time_steps, y_column):

#     gen_x=(datasets[i:i + time_steps, :-1] for i in range(len(datasets)-(time_steps + y_column)+1))
#     gen_y=(datasets[i + time_steps:i + time_steps + y_column, -1] for i in range(len(datasets)-(time_steps + y_column)+1))
#     return np.array(list(gen_x)), np.array(list(gen_y))

x,y=split_xy(datasets, 10, 1)
# print(x.shape)
# print(y)
# print(min(y),max(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)
x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, train_size=2/3, shuffle=False)

# def time_splitx(x,ts):
#     gen=(x[i:i+ts] for i in range(len(x)-ts+1))
#     return np.array(list(gen))[:,:-1]

# def time_splity(y,ts):
#     gen=(y[i:i+ts] for i in range(len(y)-ts+1))
#     return np.array(list(gen))[:,-1]

# ts=11

# x_train=time_splitx(x_train,ts)
# x_test=time_splitx(x_test,ts)
# x_predict=time_splitx(x_predict,ts)

# y_train=time_splity(y_train,ts)
# y_test=time_splity(y_test,ts)
# y_predict=time_splity(y_predict,ts)


# print(x_train)
# print(x_train.shape)
# print(type(x_train))
# print(x_test.shape)
# print(x_predict.shape)
# 2. 모델구성
model = Sequential()
model.add(Conv1D(1, 2, input_shape=(10, 13)))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.summary()

# 3. 컴파일, 훈련

# x_train = x_train.reshape(-1, 10, 13)
# # x_train = x_train.astype(np.float32)
# x_test = x_test.reshape(-1, 10, 13)
# # x_test = x_test.astype(np.float32)
# x_predict = x_predict.reshape(-1, 10, 13)
# x_predict = x_predict.astype(np.float32)

# print(x_train)
# print(x_train.shape)
# print(x_test.shape)
# print(x_predict.shape)

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)