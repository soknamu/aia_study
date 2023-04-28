import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#1. 데이터

path = './_data/wine/'
path_save = './_save/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']
#[실습] y의 클래스를 7개에서 5개로 줄여서 성능을 비교해보기!

print(train_csv['quality'].value_counts())

# for i, v in enumerate(y):
#     if v <= 5:
#         y[i] = 0
#     elif v ==6 :
#         y[i] = 1
#     else :
#         y[i] = 2

# for i, v in enumerate(y):
#     if v <= 4:
#         y[i] = 0
#     elif v ==6 |v==7 | v==5:
#         y[i] = 1
#     else :
#         y[i] = 2
    

#print(y.value_counts())

train_csv['quality'] = train_csv['quality'].replace([3, 4], 0) #3,4를 합치고 클래스의 값을 0으로 부르겠다.
train_csv['quality'] = train_csv['quality'].replace([5], 1)
train_csv['quality'] = train_csv['quality'].replace([6], 2)
train_csv['quality'] = train_csv['quality'].replace([7], 3)
train_csv['quality'] = train_csv['quality'].replace([8, 9], 4)

print(train_csv['quality'].value_counts())


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 3377, train_size= 0.7, stratify=y
    )

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성

model = RandomForestClassifier(random_state= 3377)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)

#클래스 합치기전
# 0.5290909090909091

#클래스 합친후
# 0.5466666666666666