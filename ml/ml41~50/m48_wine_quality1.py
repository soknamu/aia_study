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


#print(train_csv['quality'].value_counts())
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5

#1-1. 라벨인코더

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Check label distribution
#print(train_csv['quality'].value_counts())

# # Remove rows with single class label
# single_class_label = train_csv['quality'].nunique() == 1
# if single_class_label:
#     train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# # Split the data
# x = train_csv.drop(['quality'], axis=1)
# y = train_csv['quality']

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

# # 3.pandas get_dummies
#import pandas as pd
#y=pd.get_dummies(y)
# #print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 850, train_size= 0.7, stratify=y)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#print(x_train.shape, y_train.shape)

parameters = {'n_estimators' :1000,  
              'learning_rate' : 0.3, 
              'max_depth': 3,        
              'gamma': 0,
              'min_child_weight': 1,  
              'subsample': 0.5, 
              'colsample_bytree': 1,
              'colsample_bylevel': 1.0,
              'colsample_bynode': 1,
              'reg_alpha': 1,        
              'reg_lambda': 1,
              'random_state': 369,
              }

#print("Unique values in y_train:", np.unique(y_train))
#print("Unique values in y_test:", np.unique(y_test))


#2.모델구성

model = XGBClassifier()
model.set_params(**parameters, 
                     early_stopping_rounds=10, 
                    eval_metric='merror')
# #print(x_train.shape, y_train.shape)

model.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=True
        )

results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)
