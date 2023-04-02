#노드의 개수를 드랍한다. 
#차이점: 노드를 그냥 지우는거랑 dropout의 차이점
# 1.노드를 랜덤으로 뺀다. 
# 2.평가에서는 뺀값들도 포함해서 test측정.

#과적합을 해결하는법
#1. 신경망을 구축(훈련)할때 일부 노드를 빼고 훈련을 시킨다.
#2. 데이터가 많아야 한다. (큰데이터에는 효과가 큼) but 성능향상이 절대적이지는 않음.

# 저장할때 평가결과값, 훈련시간을 파일에 넣기. save파일 만들기.

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model #함수를 부를 때는 Model사용.
from tensorflow.python.keras.layers import Dense, Input, Dropout #인풋레이어도 명시.
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8,
                random_state= 333, shuffle=True)

# 스케일러 4가지
# scaler = MinMaxScaler() 많이 퍼저있는 것
scaler = StandardScaler() #표준분포가 모여 있으면 stand
# scaler = MaxAbsScaler() 절대값.
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_test),np.max(x_test))


#2. 모델(함수)
# input1 = Input(shape=(13,))
# dense1 = Dense(30)(input1)
# drop1 =Dropout(0.3)(dense1) #0.3프로를 드랍아웃시킴
# dense2 = Dense(20, activation= 'relu')(drop1)
# drop2 =Dropout(0.2)(dense2) #0.2프로를 드랍아웃시킴
# dense3 = Dense(10)(drop2) 
# drop3 =Dropout(0.5)(dense3) #0.5프로를 드랍아웃시킴
# output1 = Dense(1)(drop3)
# model = Model(inputs = input1, outputs = output1)

#2. 모델(Sequential)
model = Sequential()
model.add(Dense(30, input_shape=(13,)))
model.add(Dropout(0.3))
model.add(Dense(20, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')

#파일명에 시간을 넣는다.

import datetime #시간을 저장해주는 기능
date = datetime.datetime.now() #현재시간을 넣는다.
print(date) #2023-03-14 11:11:00.275117
date =date.strftime('%m%d_%H%M') #시간을 문자로 바꿔주는 코드.(파일명에 들어가게 해주기위해서)
                # 달, 일,(중간에 있는_(언더바)는 문자.) 시간, 분 반환.
print(date) #0314_1116

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#4f 는 소수점 4번째자리. 04d 몇번째 epoch 가되는지.

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint #이지점에서  save하고 말안들으면 끊어버린다.
es = EarlyStopping(monitor = 'val_loss', patience= 10, 
                   restore_best_weights=True, 
                   mode = 'min', verbose= 1) #es의 verbose의 디폴트는 0

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto',
        verbose = 1,
        save_best_only= True, #가장 좋은 점에서 적용시켜라.
        filepath= "".join([filepath, 'k27_', date, '_', filename]) 
        #빈공간.join(합친다.) -> 빈공간 + filepath(이경로에) + k27_ +date 날짜 + 파일이름.
        #파일 경로 #단축기 tip) shift tap 누르면 왼쪽으로 옮길수 있음.
                      )  #계속 지점이 낮아질수록 save함.

model.fit(x_train, y_train,epochs = 10000, 
          callbacks = [es], #,mcp 
          batch_size =32,
          verbose = 1,
          validation_split= 0.2)

#model.save('./_save/MCP/keras27_3_save_model.h5')

#세이브가 두개가 생김 mcp에서 하나, model.save에 하나.

#4. 평가 예측
from sklearn.metrics import r2_score

print('===================1. 기본 출력====================')
loss = model.evaluate(x_test, y_test,verbose = 0) #evaluate 에도 verbose 가 존재.
print('loss : ',loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score :', r2)
