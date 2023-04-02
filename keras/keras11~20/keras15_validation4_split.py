from tensorflow.python.keras.models import Sequential #파이썬을 붙이면 자동완성됨.
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size= 0.2, shuffle= True, random_state= 123)

#2. 모델구성
model = Sequential()
model.add(Dense(12,activation = 'linear', input_dim =1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs= 1, batch_size = 1, verbose = 1,
          validation_split= 0.2) # validation_data= 데이터 수정. train 데이터 수정.

print(x_test,y_test) #[ 5  2 14  3] [ 4 13  9 10]
print(x_train,y_train) #[ 5  3 13 14  9 10  4  2] [ 1 12  8  7 15 16 11  6]
#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :' , loss)

result = model.predict([15])
print('18의 예측값은 : ', result)


