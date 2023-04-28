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

train_csv.dropna(inplace=True)
test_csv.dropna(inplace=True)

#1-1. 라벨인코더

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

train_csv['quality'] = train_csv['quality'].replace([3, 4], 0) #3,4를 합치고 클래스의 값을 0으로 부르겠다.
train_csv['quality'] = train_csv['quality'].replace([5], 1)
train_csv['quality'] = train_csv['quality'].replace([6], 2)
train_csv['quality'] = train_csv['quality'].replace([7], 3)
train_csv['quality'] = train_csv['quality'].replace([8, 9], 4)

print(x.shape,y.shape)

def remove_outliers(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return clean_data

x_clean = remove_outliers(x)
y_clean = remove_outliers(y)

print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 3377, train_size= 0.7, stratify=y
    )

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

# # 4.1 내보내기
# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M%S')
# y_submit = model.predict(test_csv)

# y_submit = pd.DataFrame(y_submit)
# y_submit = np.array(y_submit)
# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# submission['quality'] = y_submit
# submission.to_csv(path_save + 'wine_' + date + '.csv')
