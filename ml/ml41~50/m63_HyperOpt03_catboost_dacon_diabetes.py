#######판다스 데이터프라임형태로 빼!!
###results 컬럼에 최소값이 있는 행을 출력!
from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

#1-1 테스트, 훈련 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337)

#1-2 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델

search_space = {
    'learning_rate' : hp.uniform('learning_rate',0.001, 0.2),
    'depth' : hp.quniform('depth',3, 8, 1),
    'one_hot_max_size' : hp.quniform('one_hot_max_size',24, 64, 1),
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf',10, 100, 1),
    'bagging_temperature' : hp.uniform('bagging_temperature',0.5, 1),
    'random_strength' : hp.uniform('random_strength', 0.5,1),
    #'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0.001,10)
    }


def cat_fuc(search_space):
    params = {
    'iterations' : 5,
    'learning_rate' : search_space['learning_rate'],
    'depth' : int(search_space['depth']),
    'one_hot_max_size' : search_space['one_hot_max_size'],
    'min_data_in_leaf' : search_space['min_data_in_leaf'],
    'bagging_temperature' : int(search_space['bagging_temperature']),
    'random_strength' : int(search_space['random_strength']),
    'task_type' : 'CPU',
    'logging_level' : 'Silent'
}
    #3. 훈련
    model = CatBoostClassifier(**params)
    
    model.fit(x_train,y_train)
    #4. 평가, 예측
    y_predict = model.predict(x_test)
    results = mean_squared_error(y_test,y_predict)
    return results
import time
start = time.time()
trial_val = Trials()
best = fmin(
    space= search_space,
    fn= cat_fuc,
    algo=tpe.suggest,
    max_evals = 50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

end = time.time()
print("best : ", best)
print("걸린시간 :", end-start)

# best :  {'bagging_temperature': 0.8199349488313555, 'depth': 8.0, 'learning_rate': 0.17096362238092064, 'min_data_in_leaf': 29.0, 'one_hot_max_size': 34.0, 'random_strength': 0.9565969305320872}
# 걸린시간 : 1.9889154434204102