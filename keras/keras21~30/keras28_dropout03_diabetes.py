from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
#1.데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442, 10) (442,) input_dim = 10



x_train, x_test, y_train, y_test = train_test_split(x,y,  #트레인과 테스트를 분리시키는 함수.
        train_size=0.7, shuffle=True, random_state=650874)
#650874

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))


#2. 모델구성

input1 = Input(shape=(10,))
dense1 = Dense(10)(input1)
dense2 = Dense(30,activation='relu')(dense1)
dense3 = Dense(50)(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(100,activation='relu')(drop1)
dense5 = Dense(50)(dense4)
dense6 = Dense(30,activation='relu')(dense5)
dense7 = Dense(5)(dense6) 
output1 = Dense(1)(dense7)
model = Model(inputs = input1, outputs = output1)


model.summary()

#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_loss', patience =250, mode = 'min',
              verbose=1,
              restore_best_weights=True) 



model.compile(loss = 'mse', optimizer= 'adam')

import time
start_time = time.time()


hist = model.fit(x_train, y_train, epochs = 5000, batch_size =32, 
                verbose = 1, validation_split= 0.2,
                callbacks= [es])

end_time = time.time()

print("===================발로스===================")
print(hist.history['val_loss'])
print("===================발로스====================")


#4. 평가예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
