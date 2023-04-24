#n_conponent > 0.95 이상
# xgboost, girdSearch 또는 RandomSearch 를 쓸것

#m33_2 결과를 뛰어넘기!

parameters = [
    {"n_estimators": [100,200,300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
    "max_depth":[4,5,6]},
    {"n_estimators": [90,100,110], "learning_rate" : [0.1, 0.001, 0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators": [90,110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6,0.7,0.9]}
]

#n_jobs = -1
    #tree_method = 'gpu_hist'
    #predictor = gpu_predictor
    #gpu_id = 0