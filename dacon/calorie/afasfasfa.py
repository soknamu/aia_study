import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
import numpy as np
# Load the dataset

#1.데이터
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

path = './_data/calorie/'
path_save = './_save/calorie/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

from sklearn.preprocessing import LabelEncoder #전처리 preprocessing
le = LabelEncoder()
le.fit(train_csv['Weight_Status']) #화이트와 레드를 0과 1로 인정하겠다.
aaa = le.transform(train_csv['Weight_Status'])

x = train_csv.drop(['Calories_Burned'],axis =1)
y = train_csv['Calories_Burned']

encoder = LabelEncoder()
df['Weight_Status'] = encoder.fit_transform(df['Weight_Status'])
df['Gender'] = encoder.fit_transform(df['Gender'])

# Split into training and testing sets
X = df.drop(['Calories'], axis=1)
y = df['Calories']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)

# Define hyperparameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

# Train the model
model = xgb.train(params, d_train, num_boost_round=100)

# Evaluate the model
y_pred = model.predict(d_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")

def RMSE(y_test, y_predict) : #함수의 약자 RMSE 임의의 값. 함수는 y= f(x)라는 식으로 계속 이용함.
   return np.sqrt(mean_squared_error(y_test,y_predict))   #리턴(나오는 값)해주면 된다. 함수를 정의 한 것. 

#np.sqrt로 루트를 씌운다.
# 여기까지는 실행이 안되고, 정의만 한 것

rmse = RMSE(y_test, y_pred)  #실행 코드 RMSE 함수 사용

print("RMSE : ",rmse)

# Save predictions to submission file
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
#submission.to_csv(path_save +'air_' + date + '.csv', index=False)

pd.DataFrame.to_csv(path_save + 'calorie_' + date + '.csv')

