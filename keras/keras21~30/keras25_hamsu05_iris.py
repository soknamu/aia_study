import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from sklearn.metrics import accuracy_score #y 의 값이 3개(0,1,2)
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
#1.데이터

datasets = load_iris()

# print(datasets.DESCR) #판다스 describe
# print(datasets.feature_names) #pandas columns inputdim =4

x = datasets.data
y = datasets['target']

#print(x.shape , y.shape) #(150, 4) (150,)
#print(x)
#print(y) #random 으로 잘해줘야됨. 데이터가 한곳에 몰려있음.


##########################이 지점에서 원핫인코딩을 한다###########################
#1. tensorflow 
# from keras.utils import to_categorical #tensorflow 빼도 가능.
# y = to_categorical(y)
# print(y.shape) #(178, 3)

# #2. sklearn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1,1)
y = ohe.fit_transform(y).toarray()
print(y.shape)

# # 3.pandas get_dummies
# import pandas as pd
# y=pd.get_dummies(y)
# print(y.shape)


x_train, x_test, y_train, y_test =  train_test_split(x,y, 
        shuffle= True, random_state= 942, train_size = 0.8,
        stratify=y)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


print(np.unique(y_train, return_counts=True)) # y_트레인의 갯수를 알려줘라.
#문제점 : 데이터가 한쪽으로 치우쳐질수도 있다. 데이터의 비율만큼 train_test+split에서 짤라줘야됨.

#print(x_train.shape, x_test.shape) #(135, 4) (15, 4)
#print(y_train.shape, y_test.shape) #(135,) (15,)

#2.모델구성

input1 = Input(shape=(4,)) #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(50,activation = 'relu')(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(40,activation = 'relu')(dense1)
dense3 = Dense(40,activation = 'relu')(dense2) 
dense4 = Dense(40,activation = 'relu')(dense3) 
output1 = Dense(3,activation = 'softmax')(dense4) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1) #인풋의 시작은 인풋1, 아웃풋의 끝은 아우풋1 

#3.컴파일

es = EarlyStopping(monitor= 'acc', patience= 50,verbose= 1, restore_best_weights= True, mode = 'max')

model.compile(loss = 'categorical_crossentropy', optimizer ='adam',
              metrics =['acc'])
model.fit(x_train, y_train, epochs =100, batch_size= 10, validation_split = 0.2, verbose =1, callbacks =[es])

# accuracy_score를 사용해서 스코어를 빼세요.
###############################################
#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
# print(y_test.shape)    #(30, 3)
# print(y_predict.shape) #(30, 3)
# print(y_test[:5])
# print(y_predict[:5]) #둘다 아그맥스 값을 때려줌

print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis = 1) #각행에 있는 열(1)끼리 비교(ytest열끼리비교)
y_predict = np.argmax(y_predict, axis = 1) #-1해도 상관없음.

print(y_predict.shape)
#print(y_test_acc.shape)

acc = accuracy_score(y_test_acc, y_predict)
print('accuary_score : ', acc)
