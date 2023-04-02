# sklearn load_wine

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score #y 의 값이 3개(0,1,2)
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터

datasets = load_digits()

# print(datasets.DESCR) #판다스 describe
# print(datasets.feature_names) #pandas columns inputdim =4

x = datasets.data
y = datasets['target']

print(x.shape , y.shape) #(1797, 64) (1797,)
#print(x)
#print(y) #random 으로 잘해줘야됨. 데이터가 한곳에 몰려있음.

print('y의 라벨값 : ', np.unique(y)) #[0 1 2 3 4 5 6 7 8 9] y의 라벨의 종류가 10가지.

##########################이 지점에서 원핫인코딩을 한다###########################

#1. tensorflow 
# from tensorflow.keras.utils import to_categorical #tensorflow 빼도 가능.
# y = to_categorical(y)
# print(y.shape) #(178, 3)

# #2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_transform(y).toarray()
# print(y.shape)

#3.pandas get_dummies
import pandas as pd
y=pd.get_dummies(y)
print(y.shape)

#1부터 시작하면 틀어질수도 있음.
#판다스에 겟더미, sklearn 원핫인코더??
################################################################################

x_train, x_test, y_train, y_test =  train_test_split(x,y, 
        shuffle= True, random_state= 942, train_size = 0.8,
        stratify=y)
#stratify = y y를 통계적으로 잘라라
print(np.unique(y_train, return_counts=True))


#print(x_train.shape, x_test.shape)
#print(y_train.shape, y_test.shape) 

#2.모델구성

model = Sequential()
model.add(Dense(60,activation = 'relu', input_dim =64))
model.add(Dense(45,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(15,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))


#3.컴파일

es = EarlyStopping(monitor= 'acc', patience= 150, restore_best_weights= True, mode = 'max')

model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
model.fit(x_train, y_train, epochs =1000, batch_size= 50, validation_split = 0.2, verbose =1, callbacks =[es])

# accuracy_score를 사용해서 스코어를 빼세요.

#4. 평가, 예측
#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)

print(y_predict.shape)
#y_test_acc = np.argmax(y_test, axis = 1) #각행에 있는 열(1)끼리 비교(ytest열끼리비교)
y_predict = np.argmax(y_predict, axis = 1) #-1해도 상관없음.

print(y_predict.shape)
#print(y_test_acc.shape)

acc = accuracy_score(y_test, y_predict)
print('accuary_score : ', acc)

# result :  [0.21652895212173462, 0.9666666388511658]
# [[0. 0. 0. ... 0. 0. 1.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 1.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# acc : 0.9666666666666667