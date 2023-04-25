# 오늘 배운 결측치처리를 마음껏 활용하여
# 성능 올려봐!

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
import pandas as pd
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer,IterativeImputer
#1.데이터
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.3 결측지 제거

#train_csv = train_csv.dropna()

# means = train_csv.mean()
# train_csv = train_csv.fillna(means)

# median = train_csv.median()
# train_csv = train_csv.fillna(median)

# train_csv = train_csv.fillna(0)

#train_csv = train_csv.fillna(method='ffill')

#train_csv = train_csv.fillna(method='bfill')

#train_csv = train_csv.fillna(value = 77)

imputer = IterativeImputer(estimator=XGBRFRegressor())
train_csv = imputer.fit_transform(train_csv)
train_csv = pd.DataFrame(train_csv)

# 1.4 x, y 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle= True, random_state=27)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

model = RandomForestRegressor(random_state=123)

#3. 훈련
model.fit(x_train,y_train)

#4. 결과
results = model.score(x_test, y_test)
print("결과는  :", results)

# dropna  결과는  : 0.7690154594697651
# mean    결과는  : 0.8064591410047406
# median  결과는  : 0.8037798579395301
# fillna0 결과는  : 0.7988195705727774
#ffill 의 결과는  : 0.7945158256049545
#bfill 의 결과는  : 0.8027306303888976
#value 77 결과는  : 0.7981843977644167

