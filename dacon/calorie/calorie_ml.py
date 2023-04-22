#소문자 함수
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
import random
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
#1.데이터
seed = 27 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

path = './_data/calorie/'
path_save = './_save/calorie/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape, test_csv.shape)

from sklearn.preprocessing import LabelEncoder #전처리 preprocessing

# Create a label encoder object
le1 = LabelEncoder()
le2 = LabelEncoder()

# Fit and transform the label column in train_csv
train_csv['Weight_Status'] = le1.fit_transform(train_csv['Weight_Status'])
train_csv['Gender'] = le2.fit_transform(train_csv['Gender'])

# Transform the label column in test_csv
test_csv['Weight_Status'] = le1.transform(test_csv['Weight_Status'])
test_csv['Gender'] = le2.transform(test_csv['Gender'])

x = train_csv.drop(['Calories_Burned'],axis =1)
y = train_csv['Calories_Burned']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=seed
)

model = make_pipeline(RobustScaler(), GaussianProcessRegressor())

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("r2_score : ", r2)

def RMSE(y_test, y_predict) : #함수의 약자 RMSE 임의의 값. 함수는 y= f(x)라는 식으로 계속 이용함.
   return np.sqrt(mean_squared_error(y_test,y_predict))   #리턴(나오는 값)해주면 된다. 함수를 정의 한 것. 

#np.sqrt로 루트를 씌운다.
# 여기까지는 실행이 안되고, 정의만 한 것

rmse = RMSE(y_test, y_predict)  #실행 코드 RMSE 함수 사용

print("RMSE : ",rmse)

# Save predictions to submission file
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
y_sub=model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]=y_sub
sample_submission_csv.to_csv(path_save + 'calorie_' + date + '.csv', index=False)
