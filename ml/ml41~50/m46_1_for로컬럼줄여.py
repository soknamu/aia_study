from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier,XGBRFRegressor,XGBClassifier,XGBRegressor
from sklearn.preprocessing import RobustScaler #Robust : 이상치에 대해서 어느정도 안정적임
import warnings
from sklearn.metrics import r2_score,mean_squared_error
#1. 데이터

x, y = load_diabetes(return_X_y=True)

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
              'random_state': 3698,
             

              }


#2. 모델
model = XGBRegressor()

for i in range(x_train.shape[1]):
    model = XGBRegressor()
    model.set_params(**parameters, 
                    early_stopping_rounds=10, 
                    eval_metric='rmse')

    x_train_selected = np.delete(x_train, i, axis=1)
    x_test_selected = np.delete(x_test, i, axis=1)

    results = 0
    r2 = 0
    mse = 0

    model.fit(x_train_selected, y_train,
            eval_set=[(x_train_selected, y_train), (x_test, y_test)],
            verbose=False)

    results = model.score(x_test_selected, y_test)
    print("최종점수 :", results)

    y_predict = model.predict(x_test_selected)
    r2 = r2_score(y_test, y_predict)
    print("r2 는", r2)

    mse = mean_squared_error(y_test, y_predict)
    print("RMSE : ", np.sqrt(mse))
    
    print(model.feature_importances_)





# [0.08517998 0.07278629 0.15219665 0.08206664 0.05656773 0.07588512
#  0.08201612 0.07183547 0.21355303 0.10791302]