from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
warnings.filterwarnings('ignore')
#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            random_state=369, train_size=0.8, stratify=y, shuffle=True)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=369)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000]  /디폴트100/ 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] / 디폴트0.3 / 0~1/ eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] /디폴트6/ 0~inf/ 정수 
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0/ 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] /디폴트 1/ 0~inf
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] /디폴트 1/ 0~inf
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 0/ 0~inf/ L1 절대값 가중치 규제/ alpha
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] /디폴트 1/ 0~inf/ L2 제곱 가중치 규제/ lambda

parameters = {'n_estimators' : [100],  # epochs 역할
              'learning_rate' : [0.3], # 학습률의 크기 너무 크면 최적의 로스값을 못잡고 너무 작으면 최소점에 가지도못하고 학습이끝남.
              'max_depth': [3],        #tree계열일때 깊이를 3개까지만 가겠다.
              'gamma': [0],
              'min_child_weight': [1], #최소의 
              'subsample': [0.5],      # dropout과 비슷한 개념.
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1],
              'reg_alpha': [1],        #규제
              'reg_lambda': [1]
              }

#L1규제 : 절대값/ 라쏘
#L2규제 : 제곱/ 리지

#2. 모델
xgb = XGBClassifier(random_state = 369)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4.평가, 예측
print("최상의 파라미터 :", model.best_params_)
print("최상의 점수 :", model.best_score_)


results = model.score(x_test, y_test)
print("최종점수 :", results)


# 최상의 매개 변수 : {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
# 최상의 점수 : 0.9494505494505494
# 최종점수 : 0.9649122807017544