#전에 했던 것과다른점 y의 값이 영과일인것 차이
#################################매우 중요#######################
#1.마지막에 시그모이드 준다.
#2.'binary_crossentropy' 를 넣어준다.
###############################################################
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
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

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2.모델구성
#중요! : 활성화 함수중에 sigmoid를 이용해서 범위를 줄여줌. (중간을 바꿔봤자 의미가 없음. 마지막만 중요.)


input1 =Input(shape=(30,))
dense1 = Dense(120, activation= 'relu')(input1)
dense2 = Dense(105,activation='relu')(dense1)
dense3 = Dense(90, activation= 'linear')(dense2)
drop1 = Dropout(0.25)(dense3)
dense4 = Dense(75,activation='relu')(drop1)
dense5 = Dense(60, activation= 'linear')(dense4)
dense6 = Dense(45,activation='relu')(dense5)
dense7 = Dense(30, activation= 'linear')(dense6) 
dense8 = Dense(15, activation= 'relu')(dense6) 
output1 = Dense(1, activation= 'sigmoid')(dense7)
model = Model(inputs = input1, outputs = output1)

es = EarlyStopping(monitor = 'val_loss', verbose= 1, patience= 150, mode = 'min', restore_best_weights= True)

#3. 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', 
              metrics=['mse', 'acc',]) #[accuracy, mean_squared_error] 

import time
start_time = time.time()

model.fit(x_train, y_train, epochs =1500, 
                 batch_size =12, verbose =1, 
                 validation_split= 0.2, callbacks = [es])

end_time = time.time()

result  =model.evaluate(x_test, y_test) #값이 여러개여서 loss 대신 result로 바뀜
print('result : ', result)

print("걸린시간  : ", round(end_time - start_time ,2))

#y_predict = model.predict(x_test)
y_predict = np.round(model.predict(x_test))


acc =accuracy_score(y_test, y_predict) #오류난 이유 y_predict을 라운드를 씌어야함.
print('acc :', acc) # acc : 0.9210526315789473
