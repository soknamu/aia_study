from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x,y = load_digits(return_X_y=True)

#1-1.트레인, 테스트 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,
            train_size=0.75, shuffle=True, random_state=1030)

#1-2 스케일러
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.fit_transform(x_test)

#1-3 kfold
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=1030, shuffle=True)

#1-4 parameters
parameters = {'n_estimators' : [100],
              'learning_rate' : [0.3],
              'max_depth': [3],
              'gamma': [0],
              'min_child_weight': [1],
              'subsample': [0.5],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1],
              'reg_alpha': [1],
              'reg_lambda': [1]
              }

#2.모델링

xgb = XGBClassifier()

model = RandomizedSearchCV(xgb, parameters, verbose=1,
                           n_jobs=-1,cv=kfold
                           )

#3. 훈련

model.fit(x_train,y_train)

#4.평가, 예측

print("최상의 파라미터 : ", model.best_params_) 
print("최상의 점수는 :", model.best_score_)

results = model.score(x_test,y_test)
print("최종점수 :", results)