#[실습]
# R2 0.55~ 0.6 이상

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
#1.데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


#print(x.shape, y.shape) #(20640, 8) (20640,)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 5)

#2. 모델구성

model = Sequential()
model.add(Dense(12,input_dim=8))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(12,activation='relu'))
model.add(Dense(21))
model.add(Dense(34))
model.add(Dense(46))
model.add(Dense(54,activation='relu'))
model.add(Dense(64))
model.add(Dense(73,activation='relu'))
model.add(Dense(83))
model.add(Dense(90))
model.add(Dense(103,activation='relu'))
model.add(Dense(118))
model.add(Dense(128))
model.add(Dense(138))
model.add(Dense(136,activation='relu'))
model.add(Dense(124))
model.add(Dense(120))
model.add(Dense(110))
model.add(Dense(95,activation='relu'))
model.add(Dense(82))
model.add(Dense(79))
model.add(Dense(62))
model.add(Dense(53,activation='relu'))
model.add(Dense(49))
model.add(Dense(38))
model.add(Dense(27))
model.add(Dense(20,activation='relu'))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam') 
hist = model.fit(x_train, y_train, epochs = 7777, batch_size =740,verbose =1, validation_split=0.2)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss']) #-> 이것을 통해서 어느지점에서 loss가 줄어들고, 늘어나는지 알수 있음, 
                               #또한 과적합 부분을 찾아서 줄일수도 있음.

plt.show()

# loss :  1.412699818611145
# r2스코어 :  -0.09029764791298245 epochs = 777


#loss :  1.2958897352218628
#r2스코어 :  -0.00014556258512499198 epochs = 713
