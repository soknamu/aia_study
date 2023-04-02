import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score #y 의 값이 3개(0,1,2)
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터

datasets = load_iris()

# print(datasets.DESCR) #판다스 describe
# print(datasets.feature_names) #pandas columns inputdim =4

x = datasets.data
y = datasets['target']

#print(x.shape , y.shape) #(150, 4) (150,)
#print(x)
#print(y) #random 으로 잘해줘야됨. 데이터가 한곳에 몰려있음.

#print('y의 라벨값 : ', np.unique(y)) #[0 1 2] y의 라벨의 종류가 3가지.

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

## y를 (150, ) -> (150,3)
#판다스에 겟더미, sklearn 원핫인코더??
################################################################################

x_train, x_test, y_train, y_test =  train_test_split(x,y, 
        shuffle= True, random_state= 942, train_size = 0.8,
        stratify=y)
#stratify = y y를 통계적으로 잘라라
print(np.unique(y_train, return_counts=True)) # y_트레인의 갯수를 알려줘라.
#문제점 : 데이터가 한쪽으로 치우쳐질수도 있다. 데이터의 비율만큼 train_test+split에서 짤라줘야됨.

#print(x_train.shape, x_test.shape) #(135, 4) (15, 4)
#print(y_train.shape, y_test.shape) #(135,) (15,)

#2.모델구성

model = Sequential()
model.add(Dense(50,activation = 'relu', input_dim = 4))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(3,activation = 'softmax')) #아웃풋을 3개뽑기 때문에 아웃풋 3개(y의 라벨값의 개수,클래스의 개수)


#라벨에 개수많큼 노드의 개수를 잡음. 0.5 0.4 0.1 = 1 그리고 제일큰놈이 아웃풋이다.
#다중분류는 'softmax' Softmax함수는 최대값 함수에 근사하도록 동작한다.
#지수를 사용하기 때문에 값이 무한대로 되는 것을 방지하기위해 최댓값에 근사하도록 동작.
# 모든 확률의 합은 1 , 그리고 거기서 제일 높은 값이 예측값.
#다중분류 문제는 마지막 레이어에는 softmax y의 라벨개수만큼 노드(output)를 정해준다.

# one hot encording : 목록화해서 개별로 그 목록값에 대한 이진값으로 만드는 방법이다. 
# 비트 들의 모임 중 하나만 1이고 나머지는 모두 0인 비트 들의 그룹을 원핫(One-hot)이라고 합니다.

# 원핫 인코딩 : 값을 매기지 말고, 위치로만 매기자.
#            0    1    2
# 가위 0     1    0    0  = 1
# 바위 1     0    1    0  = 1
# 보 2       0    0    1  = 1
#-> 가장 핫한놈 하나에게 1을 준다.
# y.shape = () , (150,3)

# 0                   1 0 0
# 1                   0 1 0
# 2         ->        0 0 1
# 0                   1 0 0
# 2                   0 0 1
# 0                   1 0 0
#(6,1)                (6,3)


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

# [[1. 0. 0.]   -> 0
#  [1. 0. 0.]   -> 0
#  [0. 1. 0.]   -> 1
#  [0. 0. 1.]   -> 2
#  [0. 1. 0.]]  -> 1
# [[0.9770373  0.0128829  0.01007981]    -> 1
#  [0.9863392  0.00791134 0.00574945]    -> 1
#  [0.03211059 0.56448054 0.40340883]    -> 2
#  [0.00260839 0.29542312 0.7019685 ]    -> 0
#  [0.0402108  0.61499083 0.3447984 ]]   -> 1
print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis = 1) #각행에 있는 열(1)끼리 비교(ytest열끼리비교)
y_predict = np.argmax(y_predict, axis = 1) #-1해도 상관없음.

print(y_predict.shape)
#print(y_test_acc.shape)

acc = accuracy_score(y_test_acc, y_predict)
print('accuary_score : ', acc)


'''
#4. 평가, 예측

result  = model.evaluate(x_test, y_test)
print('result : ', result)

# y_predict =np.argmax(model.predict(x_test),axis =1) #새로운코드 np.round 반올림.
# accuracy_score(y, np.argmax(y_predict))

# acc =accuracy_score(y_test, y_predict)       #sklearn.metrics 에서 퍼옴.
# print(y_predict)
# print('acc :', acc)


y_predict = model.predict(x_test)
print(y_predict.shape)

y_predict = np.argmax(y_predict, axis = -1) # 다시 벡터로 돌아감 (원핫인코딩 해제)
print(y_predict.shape)

y_test = np.argmax(y_test, axis = -1)  #np.argmax (axis = 1)->  1 2 3

print(y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc :', acc )
'''