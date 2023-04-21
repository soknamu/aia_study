import numpy as np
import pandas as pd
import random
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set random seed for reproducibility
seed = 27
random.seed(seed)
np.random.seed(seed)

# Load data
path = './_data/calorie/'
path_save = './_save/calorie/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# Preprocessing
le1 = LabelEncoder()
le2 = LabelEncoder()
train_csv['Weight_Status'] = le1.fit_transform(train_csv['Weight_Status'])
train_csv['Gender'] = le2.fit_transform(train_csv['Gender'])
test_csv['Weight_Status'] = le1.transform(test_csv['Weight_Status'])
test_csv['Gender'] = le2.transform(test_csv['Gender'])

# Split into train and test sets
X_train = train_csv.drop(['Calories_Burned'], axis=1)
y_train = train_csv['Calories_Burned']
X_test = test_csv.copy()

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBRegressor(random_state=seed)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
r2_train = r2_score(y_train, y_pred_train)


print('R-squared on train set: {:.4f}'.format(r2_train))
if r2_train < 0.8:
    print('Warning: R-squared on train set is less than 0.8')
y_pred = model.predict(X_test)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
submission_csv[submission_csv.columns[-1]] = y_pred
now = datetime.datetime.now().strftime('%m%d_%H%M%S')
submission_csv.to_csv(path_save + 'calorie_' + now + '.csv', index=False)
