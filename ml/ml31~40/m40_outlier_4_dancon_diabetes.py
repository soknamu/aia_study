import numpy as np
from sklearn.covariance import EllipticEnvelope
import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#1.데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'
train_csv = pd.read_csv(path +  'train.csv', index_col=0) 
test_csv = pd.read_csv(path +  'test.csv', index_col=0) 

# 1.4 x, y 분리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

imputer = IterativeImputer(estimator=XGBRFRegressor())
x = imputer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle= True, random_state=27
)

def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
    print('1사분위 :', quartile_1) 
    print('2사분위 :', q2) 
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound)| (data_out<lower_bound))
outliers_loc = outliers(x)
print('이상치의 위치 : ', list((outliers_loc)))
x[outliers_loc] = 99999999999

xgb = XGBRFRegressor()
xgb.fit(x_train, y_train)
results = xgb.score(x_test,y_test)
y_submit = xgb.predict(test_csv)
print(results)

# submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
# submission['count'] = y_submit
# submission.to_csv(path_save + 'subway.csv')