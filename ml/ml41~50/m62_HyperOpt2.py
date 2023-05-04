from hyperopt import hp, Trials, STATUS_OK, tpe, fmin
from bayes_opt import BayesianOptimization
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
warnings.filterwarnings('ignore')


#1. 데이터
x,y = load_diabetes(return_X_y=True)

#1-1 테스트, 훈련 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337)

#1-2 스케일러
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2.모델
search_space = {
    'learning_rate': hp.uniform('learning_rate',0.01, 1),
    'max_depth' : hp.quniform('max_depth', 3, 16, 1),
    'num_leaves': hp.quniform('num_leaves', 24, 64, 1),
    # 'min_child_samples' : hp.uniform('min_child_samples', 10, 200),
    # 'min_child_weight' : hp.uniform('min_child_weight', 1, 50),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    # 'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    # 'max_bin' : hp.quniform('max_bin', 10, 500 ,1),
    # 'reg_lambda' : hp.uniform('reg_lambda', 0.001, 10),
    # 'reg_alpha' : hp.uniform('reg_alpha',0.01, 50),
}
# hp.quniform(label,low, high,q) : 최소부터 최대간격 q간격
# hp.uniform(label,low,high) : 최소부터 최대까지 정규분포 간격
# hp.randint(label,upper) : 0부터 최대값 upper까지. random한 정수값
#hp.loguniform(label,low, high) : exp(uniform(low,high))값 반환/이거 역시 정규분포.

def lgb_fuc(search_space):
    
    params = {
    'n_estimators' : 1000,
    'learning_rate': search_space['learning_rate'],
    'max_depth' : int(search_space['max_depth']), #안씌우면 에러뜸.
    'num_leaves': int(search_space['num_leaves']),
    'subsample' : search_space['subsample']
 }
    #3. 훈련
    model = LGBMRegressor(**params)
    
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),
                        (x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50)
    #4. 평가, 예측
    y_predict = model.predict(x_test)
    results = mean_squared_error(y_test,y_predict)
    return results
import time
start = time.time()
trial_val = Trials()
best = fmin(
    space= search_space,
    fn= lgb_fuc,
    algo=tpe.suggest,
    max_evals = 50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

end = time.time()
print("best : ", best)
print("걸린시간 :", end-start)

# best :  {'learning_rate': 0.3689574102467296, 
#          'max_depth': 13.0, 'num_leaves': 41.0, 
#          'subsample': 0.9491620337707684}
# 걸린시간 : 0.7161068916320801