#컴퓨터 터트리기.
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
#1. 데이터

datasets =fetch_covtype()

x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(581012, 54) (581012,)

#print('y의 라벨값 : ', np.unique(y)) #[1 2 3 4 5 6 7]

#1-1. tensorflow hot encorder

# from keras.utils import to_categorical #tensorflow 빼도 가능.
# y = to_categorical(y)
# y = np.delete(y, 0, axis=1)
# print(y.shape) #(581012, 8)

#2. sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
# print(y.shape) # (581012,7)
# print(type(y)) #<class 'numpy.ndarray'>

# #3.pandas get_dummies
# import pandas as pd
# y=pd.get_dummies(y)
# print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, 
    train_size= 0.7, shuffle= True, random_state= 310, stratify= y)


#2.모델구성

model = Sequential()
model.add(Dense(50,activation = 'relu', input_dim = 54))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(7,activation = 'softmax')) #아웃풋을 3개뽑기 때문에 아웃풋 3개(y의 라벨값의 개수,클래스의 개수)

model.summary() #8111

#3.컴파일

es = EarlyStopping(monitor= 'acc', patience= 150,verbose= 1, restore_best_weights= True, mode = 'max')

model.compile(loss = 'categorical_crossentropy',
 #'categorical_crossentropy',
              optimizer ='adam',
              metrics =['acc'])
#sparse_categorical_crossentropy 

import time
start_time = time.time() # 처음 훈련시간을 저장 
model.fit(x_train, y_train, epochs =1000, 
          batch_size= 3000, validation_split = 0.2, 
          verbose =1, callbacks =[es])

end_time = time.time()

#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

print("걸린시간  : ", round(end_time - start_time ,2))
# 끝시간에서 시작시간을 차감 round(end_time - start_time,2)) -> 여기서 2는 소수점 둘째자리만


'''
y_predict = model.predict(x_test)

#print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis = 1) #각행에 있는 열(1)끼리 비교(ytest열끼리비교)
y_predict = np.argmax(y_predict, axis = 1) #-1해도 상관없음.

#print(y_predict.shape)
#print(y_test_acc.shape)

acc = accuracy_score(y_test_acc, y_predict)
print('accuary_score : ', acc)
'''

#컴퓨터 터트리기.
# loss :  0.3458685874938965
# acc :  0.8567617535591125
# 걸린시간  :  322.56