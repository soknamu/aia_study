import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, GRU
import numpy as np

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항 5가지
# print(train_csv.shape, test_csv.shape)      # (652, 9), (116, 8)
# print(train_csv.columns, test_csv.columns)
# print(train_csv.info(), test_csv.info())
# print(train_csv.describe(), test_csv.describe())
# print(type(train_csv), type(test_csv))

# 1.3 결측지 제거
train_csv = train_csv.dropna()      # 결측지 원래 없음
# print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
x = np.array(x)
x = x.reshape(-1, 4, 2)
test_csv = np.array(test_csv)
test_csv = test_csv.reshape(-1, 4, 2)
# 1.5 train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=777)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# 2. 모델구성
model = Sequential()
model.add(GRU(64, input_shape=(4, 2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = np.round(model.predict(x_test))
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
    
# 4.1 내보내기
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
y_submit = np.round(model.predict(test_csv))
submission['Outcome'] = y_submit

submission.to_csv(path_save + 'submit_0309_1240.csv')

# result :  [0.5815463066101074, 0.6989796161651611]
# acc :  0.6989795918367347

# (MinMaxScaler) 
# result :  [0.5394765734672546, 0.7295918464660645]
# acc :  0.7295918367346939

# (StandardScaler) 
# result :  [0.48839113116264343, 0.7244898080825806]
# acc :  0.7244897959183674

# (MaxAbsSclaer) 
# result :  [0.5303924083709717, 0.75]
# acc :  0.75

# (RobustScaler)
# result :  [0.4808119535446167, 0.7193877696990967]
# acc :  0.7193877551020408
