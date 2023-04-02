from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model #함수를 부를 때는 Model사용.
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
x_test = scaler.transform(x_test) #x트레인의 변환범위에 맞춰야되서 변환해준다.

print(np.min(x_test),np.max(x_test)) #0.0 ~ 1.0


model = load_model('./_save/keras26_3_save_model.h5') 
#가중치, 모델까지 저장되있음. 계속 훈련시켜도 loss값은 변하지 않음. 그래서 최고로 좋은 값이 나오면 저장.
#1번과 차이점 3번은 컴파일, 훈련까지 포함되어서 저장이됨.(이유 컴파일, 훈련 밑에다가 save를 했기 때문에)

model.summary()

#3. 컴파일 훈련

# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 10)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)