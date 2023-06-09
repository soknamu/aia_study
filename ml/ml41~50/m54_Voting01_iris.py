import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

data_list=[load_iris,
           load_breast_cancer,
           load_wine,
           load_digits]

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler()}

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler()}

model_list = [XGBClassifier(),
              LGBMClassifier(),
              RandomForestClassifier(),
              DecisionTreeClassifier()]

for d in data_list:
    x,y=d(return_X_y = True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    for s in scaler_list:
        scaler = scaler_list[s]
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        for m in model_list:
            models = [('xgb', model_list[0]), ('lgbm', model_list[1]), ('rf', model_list[2]), ('dt', model_list[3])]
            # model = VotingClassifier(estimators = models, voting = 'hard')
            model = VotingClassifier(estimators = models, voting = 'soft')
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test, y_predict)
            # print(f'데이터 : {d}, 스케일러 : {s}, 정확도 : {acc}')
            print(f'데이터 : {d.__name__}, 스케일러 : {s}, 정확도 : {acc}')