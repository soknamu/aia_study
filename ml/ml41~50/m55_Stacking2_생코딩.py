import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

#3대장.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier #연산할 필요 없는 것들을 빼버림, 잘나오는 곳 한쪽으로만 감.
from catboost import CatBoostClassifier

#1. 데이터
x,y  = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, train_size=0.8, random_state=1030
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)

#2. 모델
#2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

models = [xgb,lg,cat]
li = []
for model in models:
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    y_predict = y_predict.reshape(y_predict.shape[0],1) #y_predict = y_predict.reshape(-1,1)
    li.append(y_predict)
    
    score = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print("{0} ACC : {1:.4f}".format(class_name, score))
#print(li)
#리스타안에 넘파이라 넘파이로 변경시켜줘야됨
y_stacking_predict = np.concatenate(li, axis=1)
model = CatBoostClassifier(verbose=0)
#3. 훈련
model.fit(y_stacking_predict,y_test)

#4. 평가, 예측
print('Stacking score : ', model.score(y_stacking_predict,y_predict))

