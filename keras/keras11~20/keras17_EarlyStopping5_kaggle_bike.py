import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
#얼리스탑 (새로운 개념)
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)

#print(train_csv.isnull().sum())

x = train_csv.drop(['count','casual','registered'], axis = 1)

y = train_csv['count']

# print(x.shape) #(10886, 8)
# print(y.shape) #(6493, 0)

x_train, x_test, y_train, y_test = train_test_split(x,y,
shuffle= True, train_size= 0.7, random_state=1004)

# print(x_train.shape,x_test.shape) #(7620, 8) (3266, 8)
# print(y_train.shape,y_test.shape) #(7620,) (3266,)

#2. 모델링

model = Sequential()
model.add(Dense(150,input_dim =8,activation= (LeakyReLU(0.95))))
model.add(Dense(135,activation =(LeakyReLU(0.95))))
model.add(Dense(120, activation= (LeakyReLU(0.95))))
model.add(Dense(105,activation =(LeakyReLU(0.95))))
model.add(Dense(90, activation= 'relu'))
model.add(Dense(75,activation =(LeakyReLU(0.95))))
model.add(Dense(60, activation= (LeakyReLU(0.95))))
model.add(Dense(45,activation =(LeakyReLU(0.95))))
model.add(Dense(30, activation= (LeakyReLU(0.95))))
model.add(Dense(15, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))

#3.컴파일 훈련

es = EarlyStopping(monitor = 'val_loss', patience= 50, mode= 'min', verbose= 1, restore_best_weights=True)
# -> es로 단축기 지정, monitor : 'val_loss'를 감시한다.
#                     patience : 성능이 증가하지 않는 epoch 을 몇 번이나 허용할 것인가를 정의
#   (partience 는 다소 주관적인 기준이기 때문에 사용한 데이터와 모델의 설계에 따라 최적의 값이 바뀔 수 있다.)
# restore_best_weights :최저점(최상의 가중치)을(를) 잡은 지점에서 가중치가 저장됨.
model.compile(loss = 'mse', optimizer ='adam')
hist = model.fit(x_train,y_train, epochs = 1500, batch_size= 200, verbose =1,validation_split= 0.2,
          callbacks=[es])
# EarlyStopping에서 epochs 양과 관계가 있나? 

# epoch를 십만을 잡아도 백만을 잡아도 그전에 끝이나는데 결과치에 영향을 미치는게 없지 않나요? 

# answer : 크게 상관없다. 다만, es가 안잡히면 쭉쭉나가서 변화가 생길수도 있음.

#callbacks :과적합을 방지하기 위해서는 얼리스탑이라는 콜백함수를 사용하여 적절한 시점에 학습을 조기종료시켜야한다.
#콜백함수에서 설정한 조건을 만족하면 학습을 조기종료 시킨다. 
#callbacks 는 Earlystopping에서 쓰이는 것으로 적절한 시점에 학습을 종료시키는 함수이다.

print("===================발로스===================")
print(hist.history['val_loss'])
print("===================발로스====================")

#4.평가, 훈련

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2 score :', r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print('RMSE는 :',rmse )

y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit_0310_1847 .csv')


from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()


# #loss :  21537.232421875
# r2 score : 0.34136500699047423
# RMSE는 : 146.75567687263373

# loss :  21305.234375
# r2 score : 0.3484596090657771
# RMSE는 : 145.96313461581514

# loss :  21536.08984375
# r2 score : 0.3413998704625223
# RMSE는 : 146.75179271813138 epochs =100,000

# loss :  21400.423828125
# r2 score : 0.3455489043596254
# RMSE는 : 146.28881061656833 epochs = 1,000,000