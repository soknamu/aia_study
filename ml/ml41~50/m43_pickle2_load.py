from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
from sklearn.metrics import accuracy_score
import pickle

#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            random_state=369, train_size=0.8, stratify=y, shuffle=True)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 - 피클 불러오기
path = './_save/pickle_test/'
model = pickle.load(open(path + "m43_pickle_save.dat", 'rb')) #rb읽기 전용

#3. 평가,예측
results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("acc 는" ,acc)

