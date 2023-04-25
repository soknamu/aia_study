import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,\
    load_digits,fetch_covtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from tensorflow.keras.datasets import mnist

#1. 데이터

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True),
             fetch_covtype(return_X_y=True)]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'wine :',
                  'covtype :']

for i, v in enumerate(data_list):
    x,y = v
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=27)
    #2.모델구성
    print("=====================","변환전",data_name_list[i],"===================")
    model = RandomForestClassifier()
    #3.컴파일 훈련
    model.fit(x_train,y_train)
    #4.평가예측
    result = model.score(x_train,y_train)
    print("acc :", result)
    print("=====================","변환후",data_name_list[i],"===================")
    lda = LinearDiscriminantAnalysis() 
    x_lda = lda.fit_transform(x,y)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=27)   
    print(data_name_list[i], ':', x.shape,'->', x_lda.shape)
    # lda_EVR = lda.explained_variance_ratio_
    # cumsum = np.cumsum(lda_EVR)
    #print(cumsum)
    #2.모델구성
    model = RandomForestClassifier()
    #3.컴파일 훈련
    model.fit(x_train1,y_train1)
    #4.평가예측
    result1 = model.score(x_test1,y_test1)
    print("acc :", result1)

# for i, (x,y) in enumerate(data_list):
#     lda = LinearDiscriminantAnalysis() 
#     x_lda = lda.fit_transform(x,y)   
# print(data_name_list[i], ':', x.shape,'->', x_lda.shape)

# ===================== 변환전 iris :  ===================
# acc : 1.0
# ===================== 변환후 iris :  ===================
# iris :  : (150, 4) -> (150, 2)
# acc : 0.9
# ===================== 변환전 breast_cancer : ===================
# acc : 1.0
# ===================== 변환후 breast_cancer : ===================
# breast_cancer : : (569, 30) -> (569, 1)
# acc : 0.9649122807017544
# ===================== 변환전 digits : ===================
# acc : 1.0
# ===================== 변환후 digits : ===================
# digits : : (1797, 64) -> (1797, 9)
# acc : 0.9666666666666667
# ===================== 변환전 wine : ===================
# acc : 1.0
# ===================== 변환후 wine : ===================
# wine : : (178, 13) -> (178, 2)
# acc : 1.0
# ===================== 변환전 covtype : ===================
# acc : 1.0
# ===================== 변환후 covtype : ===================
# covtype : : (581012, 54) -> (581012, 6)
# acc : 0.9559477810383553