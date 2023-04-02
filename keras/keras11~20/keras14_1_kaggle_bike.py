#index 년도, 일자, 시간 다 중요하지만, 일단은 뺀다.
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

#1.데이터
path = './_data/kaggle_bike/' #맨뒤에/오타

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)

#결측치 제거

#print(train_csv.isnull().sum()) #결측치 없음



x = train_csv.drop(['count','casual','registered'], axis =1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 1231, train_size= 0.7
)



print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)

#2. 모델링

model = Sequential()

model.add(Dense(100,input_dim = 8))
model.add(Dense(75))
model.add(Dense(65))
model.add(Dense(50))
model.add(Dense(45))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))

#3 compile

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs =8666, batch_size = 400, verbose =3)

#4. 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('r2 score :', r2)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test,y_predict)

print('RMSE :', rmse)


# loss :  24496.9375
# 103/103 [==============================] - 0s 694us/step
# r2 score : 0.25692310358877535
# RMSE : 156.51497104523494

# loss :  24494.689453125
# 103/103 [==============================] - 0s 702us/step
# r2 score : 0.2569912749587834
# RMSE : 156.50779138133754

# loss :  23702.744140625
# 103/103 [==============================] - 0s 853us/step
# r2 score : 0.27395325184495456
# RMSE : 153.9569506233521