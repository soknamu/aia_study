
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터 

path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

print(train_csv.shape) #(1459, 10)
print(test_csv.shape) #(715, 9)

print(train_csv.isnull().sum())

train_csv = train_csv.dropna()

print(train_csv.isnull().sum())

print(train_csv.shape) #(1328, 10)


x = train_csv.drop(['count'], axis= 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, shuffle= True, random_state= 942)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv) ##test_csv도 스케일러 해줘야됨.
print(np.min(x_test), np.max(x_test))

# print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
# print(y_train.shape, y_test.shape) #(929,) (399,)

#2 모델구성

input1 =Input(shape=(9,))
dense1 = Dense(150, activation= 'relu')(input1)
dense2 = Dense(135)(dense1)
drop1 = Dropout(0.25)(dense2)
dense2 = Dense(120,activation='relu')(drop1)
dense3 = Dense(105)(dense2)
drop2 = Dropout(0.25)(dense3)
dense4 = Dense(90,activation='relu')(drop2)
dense5 = Dense(75)(dense4)
dense5 = Dense(60, activation='relu')(dense4)
drop3 = Dropout(0.25)(dense5)
dense6 = Dense(45)(drop3)
dense7 = Dense(30, activation='relu')(dense6) 
dense8 = Dense(13,)(dense6)
drop4 = Dropout(0.25)(dense8) 
output1 = Dense(1,activation= 'linear')(drop4)
model = Model(inputs = input1, outputs = output1)

import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime('%m%d_%H%M%S')
print(date)

#3. 컴파일

es = EarlyStopping(monitor = 'val_loss', patience =300, mode = 'auto',
               verbose=1, restore_best_weights=True)

model.compile(loss = 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs = 15000, batch_size =22, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es]
                )

# print("===================발로스===================")
# print(hist.history['val_loss'])
# print("===================발로스====================")p

#4 평가

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 score :', r2)
def RMSE(y_predict, y_test) :
  return np.sqrt(mean_squared_error(y_predict, y_test))
rmse = RMSE(y_predict, y_test)
print('RMSE : ', rmse)

#print(test_csv.isnull().sum())
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_submit
submission.to_csv(path_save + 'ddarung_' + date + '.csv')


# loss :  1391.3408203125
# r2 score : 0.7024590049017702
# RMSE :  37.30068178629354 드랍전


