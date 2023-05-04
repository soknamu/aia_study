#회귀로 맹그러.
#회귀데이터 올인!! for문
#scaler 6개 올인!! for문.
# random 쓰기.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer,PowerTransformer,StandardScaler
# QuantileTransformer: 분위수 1.정규분포로 만든다, 분위수로 나눈다 0에서1사이로. stand+min 성능이 좋아질수도? 데이터가 몰려있을때 유용
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

x,y = make_blobs(random_state=337,
                 n_samples=50, #샘플 50개 만듬 #가우시안 정규분포 샘플 생성.
                 centers=2,
                 cluster_std =1) 

# print(x)
# prit(y)
# print(x.shape, y.shape) #(100, 2) (100,)  -> (50, 2) (50,)

fig, ax = plt.subplots(2,2, figsize = (12,8))


ax[0,0].scatter(x[:,0], x[:,1],#모든행의 1번째 점 50개찍음.
            c=y,
            edgecolors='black' #가장자리에 검정색을 넣어라.
            ) 
ax[0,0].set_title('original')


scaler = QuantileTransformer(n_quantiles=50)
x_trans = scaler.fit_transform(x)
ax[0,1].scatter(x_trans[:,0], x_trans[:,1],#모든행의 1번째 점 50개찍음.
            c=y,
            edgecolors='black' #가장자리에 검정색을 넣어라.
            )
ax[0,1].set_title('QuantileTransformer')


scaler = PowerTransformer()
x_trans = scaler.fit_transform(x)
ax[1,0].scatter(x_trans[:,0], x_trans[:,1],#모든행의 1번째 점 50개찍음.
            c=y,
            edgecolors='black' #가장자리에 검정색을 넣어라.
            ) 
ax[1,0].set_title('PowerTransformer')


scaler = StandardScaler()
x_trans = scaler.fit_transform(x)
ax[1,1].scatter(x_trans[:,0], x_trans[:,1],#모든행의 1번째 점 50개찍음.
            c=y,
            edgecolors='black' #가장자리에 검정색을 넣어라.
            ) 
ax[1,1].set_title('StandardScaler')
plt.show()
#차이!!! [0~1]