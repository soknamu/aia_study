import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
#1. 데이터

x,y = fetch_covtype(return_X_y=True)

# train_csv['quality'] = train_csv['quality'].replace([3, 4], 0) #3,4를 합치고 클래스의 값을 0으로 부르겠다.
# train_csv['quality'] = train_csv['quality'].replace([5], 1)
# train_csv['quality'] = train_csv['quality'].replace([6], 2)
# train_csv['quality'] = train_csv['quality'].replace([7], 3)
# train_csv['quality'] = train_csv['quality'].replace([8, 9], 4)

print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 3377, train_size= 0.7, stratify=y
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성

model = RandomForestClassifier(random_state= 3377)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)

print("=================SMOTE 적용후 ============================")
smote = SMOTE(random_state=3377,k_neighbors= 3) #최근접 이웃 방식. K: n개와 같은느낌. (단점: 생성시간이 엄청 오래걸림.)
x_train, y_train = smote.fit_resample(x_train, y_train) #값이 쏠리는 것을 막기위해 사용.
print(x_train.shape,y_train.shape) #(159, 13) (159,)
print(pd.Series(y_train).value_counts().sort_index()) #웬만하면 y_test는 건드리지 않기.

#2. 모델

model = RandomForestClassifier(random_state=3377)

#3. 훈련
model.fit(x_train,y_train)

#4.평가, 예측
score = model.score(x_test, y_test)

y_predict = model.predict(x_test)

print("model_score :", score)