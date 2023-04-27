from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
from sklearn.metrics import accuracy_score
#warnings.filterwarnings('ignore')
#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            random_state=369, train_size=0.8, stratify=y, shuffle=True)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' :10000,  # epochs 역할
              'learning_rate' : 0.3, # 학습률의 크기 너무 크면 최적의 로스값을 못잡고 너무 작으면 최소점에 가지도못하고 학습이끝남.
              'max_depth': 3,        #tree계열일때 깊이를 3개까지만 가겠다.
              'gamma': 0,
              'min_child_weight': 1, #최소의 
              'subsample': 0.5,      # dropout과 비슷한 개념.
              'colsample_bytree': 1,
              'colsample_bylevel': 1., #xgboost.core.XGBoostError: Invalid Parameter format for colsample_bylevel expect float but value='[1]' 리스트형태하지마라
              'colsample_bynode': 1,
              'reg_alpha': 1,        #규제
              'reg_lambda': 1,
              'random_state': 369,
            #'eval_metric' : 'rmse'

              } #정의되지 않은 것도 들어감 하지만, 적용은 안됨.

#L1규제 : 절대값/ 라쏘
#L2규제 : 제곱/ 리지

#2. 모델
model = XGBClassifier()
#model = XGBClassifier(**parameters)

#3. 훈련
model.set_params(early_stopping_rounds=10, **parameters) #두개다 먹힘.
model.fit(x_train, y_train,
          eval_set = [(x_train,y_train),(x_test,y_test)], #validation 이랑 같음.
          # early_stopping_rounds=10,
          verbose = True,
           #eval_metric = 'logloss', #이진분류
           #eval_metric = 'error'    #이진분류
           eval_metric = 'auc'      #이진분류
           #eval_metric= 'merror'    #다중분류  mlogloss
           #eval_metric = 'rmse', 'mae', 'rmsle',.... #회귀
          )


results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("acc 는" ,acc)


####################################
# pickle.dump(model, open(path + "m43_pickle_save.dat", 'wb')) # write바이너리 : 읽기
# #model에 가중치 저장 open에 경로 파일이름 저장.

# import joblib
path = './_save/pickle_test/'
# joblib.dump(model, path + "m44_joblib1_save.dat")

model.save_model(path + "m45_xgb1_save.dat")