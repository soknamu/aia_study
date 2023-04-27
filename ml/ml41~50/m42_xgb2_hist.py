from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier
from sklearn.preprocessing import StandardScaler,RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
#warnings.filterwarnings('ignore')
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
#keras에서는 fit에서 반환하는데 여기는 fit아님.

# #4.평가, 예측
# print("최상의 파라미터 :", model.best_params_)
# print("최상의 점수 :", model.best_score_)

results = model.score(x_test, y_test)
print("최종점수 :", results)

# [30]    validation_0-rmse:0.09936 (loss)      validation_1-rmse:0.12567 (Val_loss)
# 최종점수 : 0.9824561403508771

# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=1.0, colsample_bynode=1, colsample_bytree=1,
#               early_stopping_rounds=10, enable_categorical=False,
#               eval_metric='rmse', feature_types=None, gamma=0, gpu_id=None,
#               grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.3, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=3, max_leaves=None,
#               min_child_weight=1, missing=nan, monotone_constraints=None,
#               n_estimators=10000, n_jobs=None, num_parallel_tree=None,
#               predictor=None, random_state=369, ...)

print("=========================================")
hist = model.evals_result()
#print(hist)

import matplotlib.pyplot as plt

# Get training and validation loss from model history
train_loss = hist['validation_0']['auc']
val_loss = hist['validation_1']['auc']

# Plot loss metric
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.xlabel('Number of rounds')
plt.ylabel('auc')
plt.title('auc curve')
plt.legend()
plt.show()
