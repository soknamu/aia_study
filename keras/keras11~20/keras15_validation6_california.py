#[실습]
# R2 0.55~ 0.6 이상

from sklearn.datasets import fetch_california_housing

#1.데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


#print(x.shape, y.shape) #(20640, 8) (20640,)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.7, shuffle=True, random_state= 777)

#2. 모델구성

model = Sequential()
model.add(Dense(12,input_dim=8))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(20))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. compile

model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 7, batch_size =765, validation_split= 0.2)

#4. 평가,예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)      
                                       

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)