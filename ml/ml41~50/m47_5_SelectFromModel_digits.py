from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier,XGBRegressor
from sklearn.preprocessing import RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
from sklearn.metrics import accuracy_score,mean_squared_error
#1. 데이터

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
            random_state=369, train_size=0.8, shuffle=True)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' :1000,  
              'learning_rate' : 0.3, 
              'max_depth': 3,        
              'gamma': 0,
              'min_child_weight': 1,  
              'subsample': 0.5, 
              'colsample_bytree': 1,
              'colsample_bylevel': 1.0,
              'colsample_bynode': 1,
              'reg_alpha': 1,        
              'reg_lambda': 1,
              'random_state': 369,
              }

model = XGBClassifier()
model.set_params(**parameters, 
                    early_stopping_rounds=10, 
                    eval_metric='auc')

model.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=False)

results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
r2 = accuracy_score(y_test, y_predict)
print("r2 는", r2)

mse = mean_squared_error(y_test, y_predict)
print("RMSE : ", np.sqrt(mse))

#print(model.feature_importances_)
# [0.06718344 0.07979492 0.16886368 0.0659201  0.07611872 0.07027918
#  0.09183102 0.0683997  0.21105228 0.10055698]

thresholds = np.sort(model.feature_importances_) #-> 가장 낮은 값부터 순차적으로 정리. 
#print(thresholds)
# [0.0659201  0.06718344 0.0683997  0.07027918 0.07611872 0.07979492
#  0.09183102 0.10055698 0.16886368 0.21105228]

from sklearn.feature_selection import SelectFromModel

for i in thresholds: 
    selection = SelectFromModel(model, threshold=i, #특정값 이상인 특성만 선택.
                                prefit=True,#사전 훈련된 것을 사용하겠어? False면 다시 훈련.
                               ) #t사이킷런 1.0.2는 false로 했을 때 안돌아감. 즉, True만됨.
    
    select_x_train = selection.transform(x_train)  #맨 처음 값이 먼저 들어 갔을때 10개 거치고, 두번째하는 낮은값 제외하고 돌아감.
    select_x_test = selection.transform(x_test)
    #print('변형된 x_train:', select_x_train.shape,'변형된 x_test:', select_x_test.shape)
    # 변형된 x_train: (353, 10)변형된 x_test:(89, 10)
    # 변형된 x_train: (353, 9) 변형된 x_test: (89, 9)
    # 변형된 x_train: (353, 8) 변형된 x_test: (89, 8)
    # 변형된 x_train: (353, 7) 변형된 x_test: (89, 7)
    # 변형된 x_train: (353, 6) 변형된 x_test: (89, 6)
    # 변형된 x_train: (353, 5) 변형된 x_test: (89, 5)
    # 변형된 x_train: (353, 4) 변형된 x_test: (89, 4)
    # 변형된 x_train: (353, 3) 변형된 x_test: (89, 3)
    # 변형된 x_train: (353, 2) 변형된 x_test: (89, 2)
    # 변형된 x_train: (353, 1) 변형된 x_test: (89, 1)
    
    selection_model = XGBClassifier()
    selection_model.set_params(**parameters, 
                    early_stopping_rounds=10, 
                    eval_metric='auc'
                    )

    selection_model.fit(select_x_train, y_train,
            eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
            verbose=False)

    selection_y_predict = selection_model.predict(select_x_test)
    r2 = accuracy_score(y_test, selection_y_predict)
    print("Tresh=%.3f, n=%d, R2: %.2f%%"%(i, select_x_train.shape[1], r2*100))
    #%.3F -> i %, %d(정수형) => select , %.2f%% -> r2*100  
    # mse = mean_squared_error(y_test, selection_y_predict)
    # print("RMSE : ", np.sqrt(mse))
    
# 최종점수 : 0.9666666666666667
# r2 는 0.9666666666666667
# RMSE :  0.18257418583505536
# Tresh=0.010, n=4, R2: 96.67%
# Tresh=0.168, n=3, R2: 96.67%
# Tresh=0.339, n=2, R2: 96.67%
# Tresh=0.483, n=1, R2: 96.67%
