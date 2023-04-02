#함수와 sequential와 차이점

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model #함수를 부를 때는 Model사용.
from tensorflow.python.keras.layers import Dense, Input #인풋레이어도 명시.
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
#전 처리 
# standardScaler 스탠다는 데이터를 가운데로 모아놓음.
#일요일자 과제) ,minmax 스케일러, 스탠다드 스케일러, maxabs, robust가 어떤 기능인가 설명(3~4). 
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(type(x)) # <class 'numpy.ndarray'>
# print(x)
# print(np.min(x),np.max(x)) #0.0 ~ 711.0
# scalar = MinMaxScaler()

# x = scalar.fit_transform(x)
# print(np.min(x),np.max(x)) #0.0 ~ 1.0

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8,
                random_state= 333, shuffle=True)
#스케일러는 훈련 설정전에 함.

# 스케일러 4가지
# scaler = MinMaxScaler() 많이 퍼저있는 것
scaler = StandardScaler() #표준분포가 모여 있으면 stand
# scaler = MaxAbsScaler() 절대값.
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train) #ㅌ-train에 맞춰서 바뀌어짐.
x_test = scaler.transform(x_test) #x트레인의 변환범위에 맞춰야되서 변환해준다.

print(np.min(x_test),np.max(x_test)) #0.0 ~ 1.0
#-0.00557837618540494 1.1478180091225068 범위 밖을 벗어나서 훈련이 잘된다.

# #2.모델 (함수)

model = Sequential()
model.add(Dense(30,input_shape=(13,),name = 'S1'))
model.add(Dense(20,name = 'S2'))
model.add(Dense(10,name = 'S3'))
model.add(Dense(1,name = 'S4'))

model.summary()
#함수
input1 = Input(shape=(13,),name = 'h1') #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(30,name = 'h2')(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(20,name = 'h3')(dense1)
dense3 = Dense(10,name = 'h4')(dense2) 
output1 = Dense(1,name = 'h5')(dense3) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1) #인풋의 시작은 인풋1, 아웃풋의 끝은 아우풋1 
# name = '' 이름 변경

model.summary()

#함수는 input레이어 까지 나옴, Sequential은 인풋레이어가 나오지 않음.


#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)
