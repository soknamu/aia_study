import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import time
import random
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= seed, train_size= 0.7, stratify=y
)

n_split = 5
kfold = KFold(n_splits= n_split, random_state= 56, shuffle=True)

parameters = [{'n_estimators' : [100, 200, 300]}, {'max_depth' : [6, 10, 15, 12]}, 
            {'min_samples_leaf' : [3, 10]},
    {'min_samples_split' : [2, 3, 10]}, 
    {'max_depth' : [6, 8, 12]}, 
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_estimators' : [100, 200, 400]},
    {'min_samples_split' : [2, 3, 10]},
]

#2. 모델

model = HalvingRandomSearchCV(RandomForestClassifier(),parameters,
                              cv=kfold,
                              verbose=1,
                              refit=True,
                              factor= 3.5,
                              n_jobs=3)

#3.컴파일 훈련
start = time.time()
model.fit(x_train,y_train)

print("최적의 파라미터 : ", model.best_params_)
print("최적의 점수 : ", model.best_score_)
print("model_score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("ACC : ", accuracy_score(y_test,y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
print("최적 튠ACC : ", accuracy_score(y_test,y_predict_best))

print(f'runtime : {np.round(time.time()-start,4)}')

# 최적의 파라미터 :  {'max_depth': 15}
# 최적의 점수 :  0.9333333333333333
# model_score :  0.9777777777777777
# ACC :  0.9777777777777777
# 최적 튠ACC :  0.9777777777777777
# runtime : 3.1955

# n_iterations: 2                    #2번 전체 훈련
# n_required_iterations: 3
# n_possible_iterations: 2
# min_resources_: 102.0              #최소 훈련 데이터 갯수
# max_resources_: 1257               #최대 훈련 데이터 갯수
# aggressive_elimination: False
# factor: 3.5                        #n_candidates/3.5, n_resources * 3.5
# ----------
# iter: 0
# n_candidates: 25                   #전체 파라미터 개수
# n_resources: 102                   #0번째 훈련때 쓸 훈련데이터 개수
# Fitting 5 folds for each of 25 candidates, totalling 125 fits
# ----------
# iter: 1
# n_candidates: 8                   #전체 파라미터 개수/factor
# n_resources: 357                  #min_resources * factor
# Fitting 5 folds for each of 8 candidates, totalling 40 fits
# runtime : 15.404804468154907