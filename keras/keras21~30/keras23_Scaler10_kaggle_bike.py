import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
#얼리스탑 (새로운 개념)
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
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

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv) ##test_csv도 스케일러 해줘야됨.
print(np.min(x_test), np.max(x_test))

# print(x_train.shape,x_test.shape) #(7620, 8) (3266, 8)
# print(y_train.shape,y_test.shape) #(7620,) (3266,)

#2. 모델링

model = Sequential()
model.add(Dense(150,input_dim =8,activation= 'relu'))
model.add(Dense(105,activation ='relu'))
model.add(Dense(90, activation= 'relu'))
model.add(Dense(45,activation ='relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(15, activation= 'relu'))
model.add(Dense(1, activation= 'linear'))
#model.summary() 76291

#3.컴파일 훈련

es = EarlyStopping(monitor = 'val_loss', patience= 300, mode= 'min', verbose= 1, restore_best_weights=True)

model.compile(loss = 'mse', optimizer ='adam')
hist = model.fit(x_train,y_train, epochs = 5500, batch_size= 400, verbose =1,validation_split= 0.2,
          callbacks=[es])


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
submission.to_csv(path_save + 'submit_0313_1210 .csv')


# from matplotlib import pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(hist.history['val_loss'])
# plt.subplot(1,2,2)
# plt.plot(hist.history['val_acc'])
# plt.title('val_acc')
# plt.show()


# 103/103 [==============================] - 0s 786us/step - loss: 57221.8672
# loss :  57221.8671875
# r2 score : -0.7728213145520941
# RMSE는 : 239.21092524704696 Minmaxscaler 이건 test_csv를 안해줌.

# loss :  22044.576171875
# r2 score : 0.31702517193547375
# RMSE는 : 148.47416014139552 Minmaxscaler

# loss :  21568.001953125
# r2 score : 0.331789993728392
# RMSE는 : 146.8605045702671 Stand

# loss :  21883.423828125
# r2 score : 0.3220178981123005
# RMSE는 : 147.93047205228902 Abs


# loss :  21401.533203125
# r2 score : 0.3369474803034884
# RMSE는 : 146.29264545931474 RobustScaler



