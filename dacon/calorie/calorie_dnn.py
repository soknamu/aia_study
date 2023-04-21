#소문자 함수
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import random
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout,LeakyReLU, BatchNormalization
#1.데이터
seed = 27 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

path = './_data/calorie/'
path_save = './_save/calorie/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape, test_csv.shape)

from sklearn.preprocessing import LabelEncoder #전처리 preprocessing

# Create a label encoder object
le1 = LabelEncoder()
le2 = LabelEncoder()

# Fit and transform the label column in train_csv
train_csv['Weight_Status'] = le1.fit_transform(train_csv['Weight_Status'])
train_csv['Gender'] = le2.fit_transform(train_csv['Gender'])

# Transform the label column in test_csv
test_csv['Weight_Status'] = le1.transform(test_csv['Weight_Status'])
test_csv['Gender'] = le2.transform(test_csv['Gender'])

x = train_csv.drop(['Calories_Burned'],axis =1)
y = train_csv['Calories_Burned']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=seed
)

# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=9,activation= LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation= LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(256,activation= LeakyReLU(0.55)))
model.add(Dropout(0.125))
model.add(Dense(64,activation= LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(32,activation= LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(8,activation= LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=1242, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=100000, batch_size=20, verbose=1, validation_split=0.2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#4. 평가, 예측

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2_score : ", r2)

def RMSE(y_test, y_predict) : #함수의 약자 RMSE 임의의 값. 함수는 y= f(x)라는 식으로 계속 이용함.
   return np.sqrt(mean_squared_error(y_test,y_predict))   #리턴(나오는 값)해주면 된다. 함수를 정의 한 것. 


rmse = RMSE(y_test, y_predict)  #실행 코드 RMSE 함수 사용

print("RMSE : ",rmse)

# Save predictions to submission file
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
y_sub=model.predict(test_csv)

sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]=y_sub
sample_submission_csv.to_csv(path_save + 'calorie_' + date + '.csv', index=False)
