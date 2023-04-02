from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
#1.데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 777)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_test), np.max(x_test))
#2. 모델구성

input1 = Input(shape=(8,)) #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(20)(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(25)(drop1)
dense3 = Dense(30)(dense2)
drop2 = Dropout(0.4)(dense3)
dense4 = Dense(27)(dense3)
dense5 = Dense(26)(dense4) 
drop3 = Dropout(0.2)(dense5)
dense6 = Dense(30)(dense5) 
output1 = Dense(1)(dense6) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1)
#3. 컴파일

es = EarlyStopping(monitor = 'val_loss', patience =200, mode = 'min',
              verbose=1,
              restore_best_weights=True) 

model.compile(loss = 'mse', optimizer= 'adam')

import time


hist = model.fit(x_train, y_train, epochs = 5000, batch_size =100, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es])

print("===================발로스===================")
print(hist.history['val_loss'])
print("===================발로스====================")


#4. 평가예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
