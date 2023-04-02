import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input,LeakyReLU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
#1. 데이터

path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

#print(train_csv.shape) #(5497, 13)
#print(test_csv.shape) #(1000, 12)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

#print(train_csv.isnull().sum()) #결측치 x

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
#test_csv = test_csv.drop(['type'],axis = 1)
# print(x.shape) #(5497, 11)
# print(y.shape) #(5497,)

#print( np.unique(y)) #[3 4 5 6 7 8 9] 7개

ohe = OneHotEncoder()
y = train_csv['quality'].values
y = y.reshape(-1,1)
print(type(y))
y = ohe.fit_transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 850, train_size= 0.7, stratify=y)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성

input1 = Input(shape = (12,))
dense1 = Dense(60,activation= 'relu')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(50,activation= 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense4 = Dense(40,activation= 'relu')(drop2)
drop3 = Dropout(0.4)(dense4)
dense5 = Dense(30,activation= 'relu')(drop3)
dense6 = Dense(20,activation= 'relu')(dense5)
drop4 = Dropout(0.5)(dense5)
output1 = Dense(7,activation= 'softmax')(drop4)
model = Model(inputs = input1, outputs = output1)

#3.컴파일

es = EarlyStopping(monitor= 'val_acc', patience= 550, mode = 'max',
                   restore_best_weights= True,
                   verbose= 1)



model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])

#날짜입력.
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
print(date)

filepath = './_save/MCP/dacon_wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto',
#                        verbose = 1, 
#                        save_best_only= True,
#              filepath = "".join([filepath, 'wine_', date, '_', filename]))

hist = model.fit(x_train, y_train, epochs = 6500, 
          batch_size = 70, verbose = 1,
          validation_split= 0.2,
          callbacks = [es,
                       #mcp
                       ])

#4. 평가, 예측

results = model.evaluate(x_test,y_test)
print('results :', results)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis =-1)
y_predict = np.argmax(y_predict, axis =-1)


#print(y_predict)
acc = accuracy_score(y_test_acc, y_predict)
print('Accuary score : ', acc)

#파일저장

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis = 1)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
y_submit += 3
submission['quality'] = y_submit
submission.to_csv(path_save + 'wine_' + date + '.csv')

# from matplotlib import pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(hist.history['val_loss'])
# plt.title('categorical_crossentropy')
# plt.subplot(1,2,2)
# plt.plot(hist.history['val_acc'])
# plt.title('val_acc')
# plt.show()

