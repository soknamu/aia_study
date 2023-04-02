from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler
#1. 데이터

datasets =fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(581012, 54) (581012,)

#print('y의 라벨값 : ', np.unique(y)) #[1 2 3 4 5 6 7]

#1-1. tensorflow hot encorder

from keras.utils import to_categorical #tensorflow 빼도 가능.
y = to_categorical(y)
y = np.delete(y, 0, axis=1)
print(y.shape) #(581012, 8)

# #2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_transform(y).toarray()
# print(y.shape) # (581012,7)
# print(type(y)) #<class 'numpy.ndarray'>

# #3.pandas get_dummies
# import pandas as pd
# y=pd.get_dummies(y)
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, 
    train_size= 0.7, shuffle= True, random_state= 310, stratify= y)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2.모델구성

input1 = Input(shape=(54,))
dense1 = Dense(50,activation = 'relu')(input1)
dense2 = Dense(40,activation='relu')(dense1)
dense3 = Dense(40,activation = 'relu')(dense2)
dense4 = Dense(40,activation='relu')(dense3)
output1 = Dense(7,activation = 'softmax')(dense4)
model = Model(inputs = input1, outputs = output1)

#3.컴파일

es = EarlyStopping(monitor= 'acc', patience= 50,verbose= 1, restore_best_weights= True, mode = 'max')

model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
model.fit(x_train, y_train, epochs =1000, 
          batch_size= 3000, validation_split = 0.2, verbose =1, callbacks =[es])



# accuracy_score를 사용해서 스코어를 빼세요.
###############################################
#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)

#print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis = 1) #각행에 있는 열(1)끼리 비교(ytest열끼리비교)
y_predict = np.argmax(y_predict, axis = 1) #-1해도 상관없음.

#print(y_predict.shape)
#print(y_test_acc.shape)

acc = accuracy_score(y_test_acc, y_predict)
print('accuary_score : ', acc)

# [0.30956757068634033, 0.8753499388694763]
# loss :  0.30956757068634033
# acc :  0.8753499388694763
# accuary_score :  0.8753499632825409
