from tensorflow.python.keras.models import Sequential #파이썬을 붙이면 자동완성됨.
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size= 0.625, shuffle= True, random_state= 99)
#x는 x끼리, y는 y끼리
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size= 0.5, shuffle= True, random_state= 99)

print(x_test,y_test) 
print(x_train,y_train)
print(x_val, y_val) 
#실습 :: 잘라봐!!!
# train_test_split
#10:3:3
'''
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
model.fit(x_train,y_train, epochs= 1, batch_size = 4, verbose = 1,
          validation_data=(x_val, y_val)) # validation_data= 데이터 수정. train 데이터 수정.

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :' , loss)

result = model.predict([15])
print('18의 예측값은 : ', result)
'''

