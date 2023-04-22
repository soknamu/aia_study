import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold,HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
import pandas as pd
from sklearn.preprocessing import RobustScaler
import random
# Set random seed
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                                    shuffle=True, random_state=seed)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = seed)

# Define the base models
xgb_model = XGBRegressor(n_estimators=500, 
                         max_depth=5, 
                         learning_rate=0.1, 
                         colsample_bytree=0.8, 
                         subsample=0.8)
lgbm_model = LGBMRegressor(n_estimators=500, 
                           max_depth=5, 
                           learning_rate=0.1, 
                           colsample_bytree=0.8, 
                           subsample=0.8)

from scipy.stats import randint, uniform

xgb_params = {'n_estimators': np.random.randint(100, 1000, size=10), 
              'max_depth': np.random.randint(3, 10, size=10), 
              'learning_rate': np.random.uniform(0.01, 0.3, size=10), 
              'colsample_bytree': np.random.uniform(0.5, 0.95, size=10), 
              'subsample': np.random.uniform(0.5, 0.95, size=10)}

lgbm_params = {'n_estimators': np.random.randint(100, 1000, size=10), 
               'max_depth': np.random.randint(3, 10, size=10), 
               'learning_rate': np.random.uniform(0.01, 0.3, size=10), 
               'colsample_bytree': np.random.uniform(0.5, 0.95, size=10), 
               'subsample': np.random.uniform(0.5, 0.95, size=10)}

xgb_search = HalvingRandomSearchCV(XGBRegressor(), xgb_params, 
                                   random_state=seed, 
                                   n_jobs=-1, verbose=1)
lgbm_search = HalvingRandomSearchCV(LGBMRegressor(), lgbm_params, 
                                    random_state=seed, 
                                    n_jobs=-1, 
                                    verbose=1)

estimators = [('xgb', xgb_search), ('lgbm', lgbm_search)]
stack_model = StackingRegressor(estimators=estimators,
                                cv=kfold, 
                                final_estimator=LGBMRegressor(n_estimators=1000, 
                                                              max_depth=5, 
                                                              learning_rate=0.1, 
                                                              colsample_bytree=0.8, 
                                                              subsample=0.8))

model = make_pipeline(RobustScaler(), stack_model)

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
