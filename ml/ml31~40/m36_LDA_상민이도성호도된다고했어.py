# LDA : 선형판별분석(Linear Discriminant Analysis)
# pca데이터의 방향성에 따라 선을 그음       /  :비지도학습(y값 필요없음.)
# lda각 데이터를 클래스 별로 매치를 시킨다. ㅣ : 지도학습.(y의값이 필요함.)
#컬럼의 갯수가 클래스의 개수보다 작을때
#디폴트로 돌아가나?
#성호는 y에 라운드를 씌웠다.
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes,load_digits,fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from tensorflow.keras.datasets import cifar100

#1. 데이터

# x,y = load_digits(return_X_y=True)
#x,y = load_diabetes(return_X_y=True)

#print(y) #소수점이 없음 diabetes -> 그래서 클래스로 판단.
#print(len(np.unique(y))) #214개의 값들이 있음.
x,y = fetch_california_housing(return_X_y=True)
y = np.round(y)
# #x = x.reshape()
# pca = PCA(n_components= 99)
# x_train = pca.fit_transform(x_train)
# #print(x.shape) #(150, 3) -> 컬럼축소

lda = LinearDiscriminantAnalysis() 
x_lda = lda.fit_transform(x,y)
print(x_lda.shape) #(442, 10) (20640, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 123, train_size=0.8 #디폴트
)

#2. 모델

model =RandomForestRegressor(random_state=123)

#3. 훈련
model.fit(x_train,y_train)

#4. 결과
results = model.score(x_test, y_test)
print("결과는  :", results)

#결론은 안된다. 그러나 정수형 데이터라면 가능하고,
#round를 통해서는 가능하지만, 데이터조작이라는 점때문에 권장x

