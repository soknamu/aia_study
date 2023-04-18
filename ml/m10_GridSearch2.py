import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state=953, stratify=y
)

n_splits = 5
kfold = KFold(n_splits=n_splits, random_state= 42, shuffle= True)

Parameters = [
    {"C":[1,10,100,1000], "kernel" : ['linear'], 'degree':[3,4,5]}, #12
    {"C":[1,10,100], "kernel" : ['rbf','linear'], 'gamma':[0.001, 0.0001]}, #12
    {"C":[1,10,100,1000], "kernel" : ['sigmoid'], 
     'gamma':[0.01,0.001, 0.0001],'degree':[3,4]}, #24
]     #총 48번 돈다.

#2.모델
model = GridSearchCV(SVC,Parameters,cv=kfold, verbose=1,
                     refit= True,
                     n_jobs=-1)


#3. 컴파일, 훈련

model.fit(x_train,y_train)

        #4. 평가, 예측
score = model.score(x_test, y_test)


print("최고점수 : ", max_score)
print("최고의 매개변수 : ", best_parameters)