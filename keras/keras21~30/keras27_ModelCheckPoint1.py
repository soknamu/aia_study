from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model #함수를 부를 때는 Model사용.
from tensorflow.python.keras.layers import Dense, Input #인풋레이어도 명시.
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.8,
                random_state= 333, shuffle=True)
#스케일러는 훈련 설정전에 함.

# 스케일러 4가지
# scaler = MinMaxScaler() 많이 퍼저있는 것
scaler = StandardScaler() #표준분포가 모여 있으면 stand
# scaler = MaxAbsScaler() 절대값.
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train) #x_train에 맞춰서 바뀌어짐.
x_test = scaler.transform(x_test) #x_trian의 변환범위에 맞춰야되서 변환해준다.

print(np.min(x_test),np.max(x_test)) #0.0 ~ 1.0

#2. 모델(함수)
input1 = Input(shape=(13,)) #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(30)(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2) 
output1 = Dense(1)(dense3) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1) #인풋의 시작은 인풋1, 아웃풋의 끝은 아웃풋1 
#model.save('./_save/keras26_1_save_model.h5')#모델을 저장


#3. 컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint #이지점에서  save하고 말안들으면 끊어버린다.
es = EarlyStopping(monitor = 'val_loss', patience= 10, 
                   restore_best_weights=True, mode = 'min', verbose= 1) #es의 verbose의 디폴트는 0

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto',
        verbose = 1,
        save_best_only= True, #가장 좋은 점에서 적용시켜라.
        filepath= './_save/MCP/keras27_ModelCheckPoint1.hdf5'  #파일 경로 #단축기 tip) shift tap 누르면 왼쪽으로 옮길수 있음.
                      )  #계속 지점이 낮아질수록 save함.

model.fit(x_train, y_train,epochs = 10000, 
          callbacks = [es, mcp],
          validation_split= 0.2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)


# Epoch 00018: val_loss improved from 21.03920 to 16.78535, saving model to ./_save/MCP\keras27_ModelCheckPoint1.hdf5
# Epoch 19/10000
# 11/11 [==============================] - 0s 3ms/step - loss: 25.5069 - val_loss: 17.4260

#개선 되다가 개선이 안됨.

# Epoch 00019: val_loss did not improve from 16.78535
# Epoch 20/10000
# 11/11 [==============================] - 0s 3ms/step - loss: 26.1647 - val_loss: 19.1027