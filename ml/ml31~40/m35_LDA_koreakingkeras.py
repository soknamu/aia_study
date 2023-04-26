# LDA : 선형판별분석(Linear Discriminant Analysis)
# pca데이터의 방향성에 따라 선을 그음       /  :비지도학습(y값 필요없음.)
# lda각 데이터를 클래스 별로 매치를 시킨다. ㅣ : 지도학습.(y의값이 필요함.)
#컬럼의 갯수가 클래스의 개수보다 작을때
#디폴트로 돌아가나?

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes,load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from tensorflow.keras.datasets import cifar100

#1. 데이터

# x,y = load_digits(return_X_y=True)


(x_train,y_train), (x_test,y_test) = cifar100.load_data()

#print(x_train.shape) #(50000, 32, 32, 3)

x_train = x_train.reshape(50000, 32*32*3)
pca = PCA(n_components= 99)
x_train = pca.fit_transform(x_train)
#print(x.shape) #(150, 3) -> 컬럼축소

lda = LinearDiscriminantAnalysis() 
x_lda = lda.fit_transform(x_train,y_train)
print(x_lda.shape) #(50000, 99)
