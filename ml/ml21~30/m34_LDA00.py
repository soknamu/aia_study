# LDA : 선형판별분석(Linear Discriminant Analysis)
# pca데이터의 방향성에 따라 선을 그음       /  :비지도학습(y값 필요없음.)
# lda각 데이터를 클래스 별로 매치를 시킨다. ㅣ : 지도학습.(y의값이 필요함.)

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes,load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

#1. 데이터

x,y = load_digits(return_X_y=True)

# pca = PCA(n_components= 3)
# x = pca.fit_transform(x)
# print(x.shape) #(150, 3) -> 컬럼축소

lda = LinearDiscriminantAnalysis(n_components=3) #lda는 클래스보다 작게 나와야됨.  즉, 클래스 개수-1개 만큼나옴
x = lda.fit_transform(x,y)
print(x.shape) #fit() missing 1 required positional argument: 'y' 지도학습이라 y가필요. y의 클래스를 알고싶다.
# lda가 pca보다 더 좋은 경우가 있음 (150, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 123, train_size=0.8 #디폴트
)

#2. 모델

model =RandomForestClassifier(random_state=123)

#3. 훈련
model.fit(x_train,y_train)

#4. 결과
results = model.score(x_test, y_test)
print("결과는  :", results)

import matplotlib.pyplot as plt

#scatter plot 그리기
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.xlable('petal length')
# plt.xlable('petal width')
# plt.title('iris scatter plot')
# plt.show()