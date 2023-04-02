import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LeakyReLU, Input, Dropout
from sklearn.metrics import r2_score, f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/call/'
path_save = './_save/call/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# a = train_csv['음성사서함이용']
# print(a.shape)
# for i in range(len(a)):
#     if a[i] >= 2:
#         a[i] = 2

# train_csv['음성사서함이용'] = a
# print(train_csv)

# 1.2 확인사항 5가지
# print(train_csv.shape, test_csv.shape)
# print(train_csv.columns, test_csv.columns)
# print(train_csv.info(), test_csv.info())
# print(train_csv.describe(), test_csv.describe())
# print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)

# print(train_csv.isnull().sum())
# filled_train = imputer.fit_transform(train_csv)      ### KNNimputer 를 이용한 결측지 제거 ###
# print(filled_train.shape)      #(1328, 10) [dropna()] -> (1459, 10) [KNNimputer] : 값 채워넣기로 행 수가 유지됨

# print(filled_train)
# print(type(filled_train))      # <class 'numpy.ndarray'> : 다시 데이터 프레임으로 변환 필요

# filled_train = pd.DataFrame(filled_train, columns=train_csv.columns)
# print(type(filled_train))      # <class 'pandas.core.frame.DataFrame'>

# print(train_csv.isnull().sum())
# train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['전화해지여부'], axis=1)
# print(x)

y = train_csv['전화해지여부']
# print(y)

# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=12345, shuffle=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 2. 모델구성
input1 = Input(shape=(12,))
dense1 = Dense(32, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(16, activation='relu')(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=128, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
    
x1 = model.predict(x_test)
x1 = x1.reshape(-1,)

for i in range(len(x1)):
    if x1[i] < 0.15:
        x1[i] = 0
    else:
        x1[i] = 1

y_predict = x1
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

f1 = f1_score(y_test,y_predict, average = 'macro')
print('f1 score : ', f1)


# print(f'loss : {loss} \nrmse : {rmse}')

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x2 = model.predict(test_csv)
print(x2.shape)
x2 = x2.reshape(-1,)
print(x2.shape)
print(type(x2))
for i in range(len(x2)):
    if x2[i] < 0.15:
        x2[i] = 0
    else:
        x2[i] = 1

y_submit = x2
submission['전화해지여부'] = y_submit

submission.to_csv(path_save + 'submit_' + date + '.csv')
print(np.unique(y_submit, return_counts=True))

# import matplotlib.pyplot as plt

# plt.plot(hist.history['val_acc'],label='val_acc')
# plt.plot(hist.history['acc'],label='acc')
# plt.legend()
# plt.show()
