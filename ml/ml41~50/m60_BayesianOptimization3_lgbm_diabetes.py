from bayes_opt import BayesianOptimization
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import warnings
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
bayesian_params = {
    'learning_rate': (0.01, 0.3),
    'max_depth' : (3, 16),
    'num_leaves': (64, 256),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def lgb_fuc(learning_rate,max_depth, num_leaves, 
            min_child_samples,min_child_weight,
            subsample,colsample_bytree,
            max_bin,reg_lambda,reg_alpha):
    
    params = {
    'n_estimators' : 1000,
    'learning_rate': learning_rate,
    'max_depth' : int(round(max_depth)), #안씌우면 에러뜸.
    'num_leaves': int(round(num_leaves)),
    'min_child_samples' : int(round(min_child_samples)),
    'min_child_weight' : min_child_weight, #int round 안씌어도됨. 왜냐면 실수값이기 때문에.
    'subsample' : max(min(subsample, 1),0),#dropout과 비슷한개념.(이번 훈련을 시킬때 샘플로 하겠다.)
                                           # 0과 1의 사이.
    'colsample_bytree' : colsample_bytree,
    'max_bin' : max(int(round(max_bin)), 10), #무조건 10이상.
    'reg_lambda' : max(reg_lambda,0), #마이너스일 경우 0이상으로만 해주면 됨. 
                                      #만약에 -0.001이 최대값이면 0으로 변환해줌.
    'reg_alpha' : reg_alpha
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
    results = r2_score(y_test,y_predict)
    return results

lgb_bo = (BayesianOptimization(f=lgb_fuc,
                               pbounds=bayesian_params,
                               random_state=337))

import time
start = time.time()
n_iter = 500
lgb_bo.maximize(init_points=5, #초기점 5개 잡음. 그리고 100번돔 =105번
                n_iter=n_iter) #큰 차이 없이 105개 돔
end = time.time()
print(n_iter, "걸린시간 : ", end-start)
print(lgb_bo.max)