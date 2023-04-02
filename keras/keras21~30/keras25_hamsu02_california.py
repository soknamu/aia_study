from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Input
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
dense1 = Dense(12)(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(7)(dense1)
dense3 = Dense(8)(dense2) 
dense4 = Dense(12)(dense3) 
dense5 = Dense(21)(dense4) 
dense6 = Dense(34)(dense5) 
dense7 = Dense(46)(dense6)
dense8 = Dense(54)(dense7) 
dense9 = Dense(64)(dense8) 
dense10 = Dense(73)(dense9) 
dense11 = Dense(20)(dense10) 
dense12 = Dense(8)(dense11) 
dense13 = Dense(4)(dense12) 
output1 = Dense(1)(dense13) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1)
#3. 컴파일

es = EarlyStopping(monitor = 'val_loss', patience =20, mode = 'min',
              verbose=1,
              restore_best_weights=True) 

model.compile(loss = 'mse', optimizer= 'adam')

import time


hist = model.fit(x_train, y_train, epochs = 5000, batch_size =32, 
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

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize =(9,6))
# #plt.plot(y) #-> x는 순서대로 가기때문에 x는 명시안해도됨
# plt.plot(hist.history['loss'], marker = '.', c='red',label = 'loss') #-> 이것을 통해서 어느지점에서 loss가 줄어들고, 늘어나는지 알수 있음, 
# plt.plot(hist.history['val_loss'], marker = '.', c='blue',label = 'val_loss') 
#                                #또한 과적합 부분을 찾아서 줄일수도 있음.
# plt.title('캘리포니아') #표제목
# plt.xlabel('epochs') # x축 
# plt.ylabel('로스, 검증로스') #y축
# plt.legend() #범례표시 : 오른쪽 위에 뜨는 
# plt.grid()  # 그래프에 오목판처럼 축이 생김
# plt.show()


# loss :  1.2056472301483154
# r2 score :  0.06950199104992483