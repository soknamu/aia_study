import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#데이터 

path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

print(train_csv.shape) #(1459, 10)
print(test_csv.shape) #(715, 9)

print(train_csv.isnull().sum())

train_csv = train_csv.dropna()

print(train_csv.isnull().sum())

print(train_csv.shape) #(1328, 10)


x = train_csv.drop(['count'], axis= 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= 0.7, shuffle= True, random_state= 942)

print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

#2 모델구성

model = Sequential()
model.add(Dense(28, input_dim=9))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(17))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

#3컴파일

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =5500, batch_size = 32, verbose = 3, validation_split = 0.2)

#4 평가

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)

r2 = r2_score(y_predict, y_test)

print('r2 score :', r2)


def RMSE(y_predict, y_test) :
  return np.sqrt(mean_squared_error(y_predict, y_test))

rmse = RMSE(y_predict, y_test)

print('RMSE : ', rmse)

#print(test_csv.isnull().sum())
y_submit = model.predict(test_csv)
#print(y_submit) 

submission = pd.read_csv(path + 'submission.csv', index_col = 0)
#print(submission)

submission['count'] = y_submit
#print(submission)

submission.to_csv(path_save + 'submit_0307_0948_18 .csv')