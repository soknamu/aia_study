import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LeakyReLU, Input
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


#model.summary() 76291
input1 = Input(shape=(8,)) #인풋명시, 그리고 이걸 인풋1이라고 이름을 지정.
dense1 = Dense(150,activation = 'relu')(input1) #Dense 모델을 구성하고, 마지막은 시작은 어디에서 시작해서 끝은 어디로 끝내는지 연결해줌.
dense2 = Dense(105,activation = 'relu')(dense1)
dense3 = Dense(90,activation = 'relu')(dense2) 
dense4 = Dense(45,activation = 'relu')(dense3) 
dense5 = Dense(30,activation = 'relu')(dense4) 
dense6 = Dense(15,activation = 'relu')(dense5) 
output1 = Dense(1,activation = 'linear')(dense6) #인풋레이어는 dense1으로 dense1은 dense2로 output에서 반복...
model = Model(inputs = input1, outputs = output1)
#3.컴파일 훈련

es = EarlyStopping(monitor = 'val_loss', patience= 700, mode= 'min', verbose= 1, restore_best_weights=True)

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
submission.to_csv(path_save + 'submit_0313_1509 .csv')


# from matplotlib import pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(hist.history['val_loss'])
# plt.subplot(1,2,2)
# plt.plot(hist.history['val_acc'])
# plt.title('val_acc')
# plt.show()

# 103/103 [==============================] - 0s 605us/step - loss: 21268.3594
# loss :  21268.359375
# r2 score : 0.34107352760641296
# RMSE는 : 145.8367597428715

# loss :  21013.669921875
# r2 score : 0.3489642278715307
# RMSE는 : 144.96092603071014