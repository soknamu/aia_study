import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import pandas as pd

# Set random seed
np.random.seed(27)

# Load data
path = './_data/calorie/'
path_save = './_save/calorie/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
train_csv['Weight_Status'] = le1.fit_transform(train_csv['Weight_Status'])
train_csv['Gender'] = le2.fit_transform(train_csv['Gender'])
test_csv['Weight_Status'] = le1.transform(test_csv['Weight_Status'])
test_csv['Gender'] = le2.transform(test_csv['Gender'])

# Split data into training and testing sets
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=27)

# Define the model pipeline
model = make_pipeline(RobustScaler(), XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, colsample_bytree=0.8, subsample=0.8))

# Train the model
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("R2 score: ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE: ", rmse)

# Generate predictions on test data and save to submission file
import datetime
date = datetime.datetime.now().strftime('%m%d_%H%M%S')
y_sub = model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]] = y_sub
sample_submission_csv.to_csv(path_save + 'calorie_' + date + '.csv', index=False)
