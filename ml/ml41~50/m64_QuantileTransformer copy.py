#회귀로 맹그러.
#회귀데이터 올인!! for문
#scaler 6개 올인!! for문.
# random 쓰기.


from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype #분류.
from sklearn.datasets import fetch_california_housing, load_diabetes #회귀.
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
# QuantileTransformer: 분위수 1.정규분포로 만든다, 분위수로 나눈다 0에서1사이로. stand+min 성능이 좋아질수도? 데이터가 몰려있을때 유용
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
#1. 데이터

data_list =  [load_iris(return_X_y=True),
              load_breast_cancer(return_X_y=True),
              load_wine(return_X_y=True),
              load_digits(return_X_y=True),
              fetch_covtype(return_X_y=True)]


scaler_list = [StandardScaler(),MaxAbsScaler(),MinMaxScaler(),
               RobustScaler(),QuantileTransformer(),PowerTransformer()
               ]


for i, data in enumerate(data_list):
    x,y = data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y
                                                    )
    for s, v in enumerate(scaler_list):
        scaler = v()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        #2. 모델
        model = RandomForestClassifier()

        #3. 훈련
        model.fit(x_train, y_train)

        #4.평가, 예측
        print(type(data).__name__,  {scaler[0].__class__.__name__})
        #print(f"데이터: {v.__name__}, 스케일러 : {v2}, 점수 : " ,round(model.score(x_test,y_test)))


# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()