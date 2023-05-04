#######판다스 데이터프라임형태로 빼!!
###results 컬럼에 최소값이 있는 행을 출력!
from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from sklearn.datasets import fetch_covtype
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

#1. 데이터
x,y = fetch_covtype(return_X_y=True)

#1-1 테스트, 훈련 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337)

#1-2 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
from hyperopt import fmin, tpe, hp, Trials

def cat_fuc(params):
    model = CatBoostClassifier(bootstrap_type='Poisson',
                           subsample=params['subsample'],
                           learning_rate=params['learning_rate'],
                           max_depth=int(params['max_depth']),
                           iterations=int(params['iterations']),
                           eval_metric='Accuracy',
                           random_seed=42,
                           silent=True,
                           task_type='GPU')

    model.fit(x_train, y_train,
              eval_set=(x_test, y_test),
              use_best_model=True,
              verbose=False)
    acc = model.score(x_test, y_test)
    return {'loss': acc, 'status': STATUS_OK}

space = {
    'subsample': hp.uniform('subsample', 0.5, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'iterations': hp.quniform('iterations', 15, 30, 1)
}

trials = Trials()
best = fmin(cat_fuc, space, algo=tpe.suggest, max_evals=100, trials=trials)
print(best)

print("best : ", best)

#best :  {'iterations': 23.0, 'learning_rate': 0.8742750039747154, 'max_depth': 6.0, 'subsample': 0.7184701222469669}