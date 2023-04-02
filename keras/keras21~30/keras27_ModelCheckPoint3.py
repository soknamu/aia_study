# save_model 과 비교
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model #함수를 부를 때는 Model사용.
from tensorflow.python.keras.layers import Dense, Input #인풋레이어도 명시.
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
input1 = Input(shape=(13,)) #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(30)(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2) 
output1 = Dense(1)(dense3) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1) #인풋의 시작은 인풋1, 아웃풋의 끝은 아웃풋1 



#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint #이지점에서  save하고 말안들으면 끊어버린다.
es = EarlyStopping(monitor = 'val_loss', patience= 10, 
                   #restore_best_weights=True, 
                   mode = 'min', verbose= 1) #es의 verbose의 디폴트는 0

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto',
        verbose = 1,
        save_best_only= True, #가장 좋은 점에서 적용시켜라.
        filepath= './_save/MCP/keras27_3_MCP.hdf5'  #파일 경로 #단축기 tip) shift tap 누르면 왼쪽으로 옮길수 있음.
                      )  #계속 지점이 낮아질수록 save함.

model.fit(x_train, y_train,epochs = 10000, 
          callbacks = [es, mcp], batch_size =32,
          verbose = 1,
          validation_split= 0.2)

model.save('./_save/MCP/keras27_3_save_model.h5')

#세이브가 두개가 생김 mcp에서 하나, model.save에 하나.

#4. 평가 예측
from sklearn.metrics import r2_score

print('===================1. 기본 출력====================')
loss = model.evaluate(x_test, y_test,verbose = 0) #evaluate 에도 verbose 가 존재.
print('loss : ',loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score :', r2)

print('===================2. load_model 출력====================')
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')

loss = model2.evaluate(x_test, y_test,verbose = 0)
print('loss : ',loss)
y_predict = model2.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score :', r2)

#1번과 2번과 값 동일.

print('===================3. MCP 출력====================')
model3 = load_model('./_save/MCP/keras27_3_MCP.hdf5')

loss = model3.evaluate(x_test, y_test,verbose = 0)
print('loss : ',loss)
y_predict = model3.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score :', r2)

#1번과 2번 3번과 값 동일.
#restore  를 안키면 결과가 달라짐.

# ===================1. 기본 출력====================
# 4/4 [==============================] - 0s 661us/step - loss: 24.9220
# loss :  24.921972274780273
# r2 score : 0.7458989019097497
# ===================2. load_model 출력====================
# 4/4 [==============================] - 0s 847us/step - loss: 24.9220
# loss :  24.921972274780273
# r2 score : 0.7458989019097497
# ===================3. MCP 출력====================
# 4/4 [==============================] - 0s 1ms/step - loss: 23.6983
# loss :  23.69828224182129
# r2 score : 0.7583755051971308