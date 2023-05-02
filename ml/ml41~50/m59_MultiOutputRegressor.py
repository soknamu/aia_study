#
import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier,MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
x,y = load_linnerud(return_X_y=True)
print(x.shape) #(20, 3)
print(y.shape) #(20, 3)

model = Ridge()
model.fit(x,y)
y_pred = model.predict(x)
print("스코어는 : ",mean_absolute_error(x,y)) #스코어는 :  0.29687777631731227

# 예상 [2,110,43]  ->  원래 [138. 33. 68.]
print(model.predict([[2,110,43]]))

# model = XGBRegressor()
# model.fit(x,y)
# print("스코어는 : ",model.score(x,y)) #스코어는 :  0.29687777631731227

# # 예상 [2,110,43]  ->  원래 [138. 33. 68.]
# print(model.predict([[2,110,43]]))
# (20, 3)
# (20, 3)
# 스코어는 :  0.29687777631731227
# [[187.32842123  37.0873515   55.40215097]]
# 스코어는 :  0.9999999567184008
# [[138.00215   33.001656  67.99831 ]]

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x,y)
y_pred = model.predict(x)
print("스코어는 : ",round(mean_absolute_error(x,y),4)) #스코어는 :  0.29687777631731227
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# 예상 [2,110,43]  ->  원래 [138. 33. 68.]
print(model.predict([[2,110,43]]))
#lightgbm은 안됨. 1차원에서만 가능.
#1. 해결책 3번을 해준다
#2. MultiOutputRegressor을 쓴다.