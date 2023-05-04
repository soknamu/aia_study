import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin, tpe
from functools import partial
import warnings
warnings.filterwarnings('ignore')

#1 데이터
ddarung_path = 'c:/_study/_data/_dacon_ddarung/'
kaggle_bike_path = 'c:/_study/_data/_kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1)
y1 = ddarung['count']

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1)
y2 = kaggle_bike['count']

data_list = {'california' : fetch_california_housing, # 회귀
             'ddarung' : (x1, y1), # 회귀
             'kaggle_bike' : (x2, y2)} # 회귀

scaler_list = {'MinMax' : MinMaxScaler(),
               'Max' : MaxAbsScaler(),
               'Standard' : StandardScaler(),
               'Robust' : RobustScaler()}

for d in data_list:
    if d == 'california':
        x, y = data_list[d](return_X_y = True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    elif d == 'ddarung' or d == 'kaggle_bike':
        x, y = data_list[d]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    for s in scaler_list:
        scaler = scaler_list[s]
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        eval_metric = 'rmse'
        def cat_function (cat_hyper_params):
            model = CatBoostRegressor(max_depth = int(cat_hyper_params['max_depth']), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
                                    learning_rate  = cat_hyper_params['learning_rate'],
                                    n_estimators  = int(cat_hyper_params['n_estimators']), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
                                    min_child_weight = cat_hyper_params['min_child_weight'],
                                    subsample = cat_hyper_params['subsample'],
                                    colsample_bytree = cat_hyper_params['colsample_bytree'],
                                    max_bin = int(cat_hyper_params['max_bin']), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
                                    reg_lambda = cat_hyper_params['reg_lambda'],
                                    reg_alpha = cat_hyper_params['reg_alpha'],
                                    # objective=objective,
                                    # use_label_encoder=False,
                                    random_state = 1234)
            model.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], eval_metric = eval_metric, verbose = 0, early_stopping_rounds=50)
            y_predict = model.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_predict))
            return rmse
        cat_hyper_params = {'max_depth' : hp.quniform('max_depth', 3, 16, 1),
                            'learning_rate' : hp.uniform('learning_rate', 0.3, 0.7),
                            'n_estimators' : hp.quniform('n_estimators', 100, 500, 1),
                            'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
                            'subsample' : hp.uniform('subsample', 0.5, 1),
                            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                            'max_bin' : hp.quniform('max_bin', 10, 500, 1),
                            'reg_lambda' : hp.uniform('reg_lambda', 0.001, 10),
                            'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50)}

        cat_objective = partial(cat_function, x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, eval_metric = eval_metric)
        
        best = fmin(fn = cat_function, space = cat_hyper_params, algo = tpe.suggest, max_evals = 60, rstate = np.random.default_rng(seed=1234))
        
        best_loss = cat_function(best)

        print(f'데이터: {d}, 스케일러 : {s}, 최소 점수 : {-best_loss}')