import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan,  8, np.nan]]
                    ).transpose()


data.columns = ['x1','x2','x3','x4']
#print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer,IterativeImputer #결측치에대한 책임을 전가하겠다.

#imputer = SimpleImputer() # 디폴트 평균값.
#imputer = SimpleImputer(strategy='mean') #평균.
#imputer = SimpleImputer(strategy='median') #중의값
#imputer = SimpleImputer(strategy='most_frequent') 
#최빈값 가장 많이 나온 값으로 결측치 채움.
#만약에 다 개수가 똑같이 나오면 가장 작은 값이 나옴.
#imputer = SimpleImputer(strategy='constant', fill_value= 7777) #끊임없이
#imputer = KNNImputer()
imputer = IterativeImputer(estimator=XGBRFRegressor())

data2 = imputer.fit_transform(data)

print(data2) # 디폴트 평균값.
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

# DecisionTreeRegressor

# [[ 2.  2.  2.  4.]
#  [ 2.  4.  4.  4.]
#  [ 6.  4.  6.  4.]
#  [ 8.  8.  8.  8.]
#  [10.  8. 10.  8.]]

#XGBRFRegressor

# [[ 2.          2.          2.          4.63995981]
#  [ 5.63994408  4.          4.          4.        ]
#  [ 6.          5.11995745  6.          4.63995981]
#  [ 8.          8.          8.          8.        ]
#  [10.          6.83994198 10.          7.31993771]]