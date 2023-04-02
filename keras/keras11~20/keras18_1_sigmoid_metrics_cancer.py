#전에 했던 것과다른점 y의 값이 영과일인것 차이
#################################매우 중요#######################
#1.마지막에 시그모이드 준다.
#2.'binary_crossentropy' 를 넣어준다.
###############################################################
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
#1. data

datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR) #pandas :.describe()
#print(datasets.feature_names) # : .columns()

x = datasets['data']#딕셔너리의 key
y = datasets.target

#print(x.shape) #(569, 30)
#print(y.shape) #(569,) 벡터여서 output_dim = 1

#print(y) #암에 걸렸다.(1) 안걸렸다.(0)

x_train, x_test, y_train, y_test = train_test_split(x,y, 
        test_size= 0.2, shuffle= True, random_state= 1008)

#2.모델구성
#중요! : 활성화 함수중에 sigmoid를 이용해서 범위를 줄여줌. (중간을 바꿔봤자 의미가 없음. 마지막만 중요.)
model = Sequential()
model.add(Dense(120, input_dim =30, activation= 'relu'))
model.add(Dense(105, activation= 'relu'))
model.add(Dense(90, activation= 'linear'))
model.add(Dense(75, activation= 'relu'))
model.add(Dense(60, activation= 'linear'))
model.add(Dense(45, activation= 'relu'))
model.add(Dense(30, activation= 'linear'))
model.add(Dense(15, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid')) 


es = EarlyStopping(monitor = 'val_loss', verbose= 1, patience= 150, mode = 'min', restore_best_weights= True)

#3. 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics=['mse', 'acc',]) #[accuracy, mean_squared_error] 
model.fit(x_train, y_train, epochs =1500, 
                 batch_size =8, verbose =1, 
                 validation_split= 0.2, callbacks = [es])
#두개이상은 list 
#'binary_crossentropy' 는 나머지 소수들을 처리해줌. ex) 1.48328483 이면 0.48328483을 지워줌.
#metrics 지표를 보여줌. 계산에 영향을 안줌. 'acc' = 'accuracy' 랑 같은코드(풀네임, 약자)
#4. 평가, 예측

result  =model.evaluate(x_test, y_test) #값이 여러개여서 loss 대신 result로 바뀜
print('result : ', result)

#y_predict = model.predict(x_test)
y_predict = np.round(model.predict(x_test))

# 오류 측정
# print("==========================")
# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))
# print("==========================")

# ValueError: Classification metrics can't handle a mix of binary and continuous targets
# 에러 : 분류 메트리스는 mix of binary and continuous targets(0,1,0,1,0)를 다룰수 없다.

# [[ 0.0496819 ]
#  [ 0.9761555 ]
#  [ 0.98726296]
#  [ 0.78787446]
#  [-0.50914514]]
# 맞다 틀리다[2진분류,그외는 다중분류](1과0)로 나와아하는데 predict값이 실수값(0.99999이런식)으로 나와서 오류가 뜨는 것. 
# 즉, 아웃풋이 실수값이기 때문에 에러가 뜬다. 0과1로 범위를 줄여줘야함.
# mse 를 쓰면 안된다. 그래서 실수값을 반올림 해주는 코드를 써야됨. 그게아마sigmoid인것 같다.
#미분가능하다 = 역전파
# 계단함수의 차이점 디테일이 떨어진다.

# [0 1 1 1 0]
# [[2.1742001e-04]   [[0.]
#  [9.8471159e-01]    [1.]
#  [9.6786451e-01]    [1.]
#  [9.4301641e-01]    [1.]
#  [1.1840149e-11]]  [0.]]


acc =accuracy_score(y_test, y_predict) #오류난 이유 y_predict을 라운드를 씌어야함.
print('acc :', acc) # acc : 0.9210526315789473

# y_test   0 1 0 1
#y_predict 0 0 1 0
#          0 x x x   정확도 25%
#metrics = ['accurary'] 를 적으면 바로나옴. print안적어도됨.