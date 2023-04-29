import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier
import time

# Load data
train = pd.read_csv('c:/study/_data/dacon_airplane/train.csv')
test = pd.read_csv('c:/study/_data/dacon_airplane/test.csv')
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col=0)

# Define the function to replace outliers with NaN
def replace_outliers_with_nan(data, column, threshold):
    # Calculate z-scores for the column
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())

    # Replace values greater than the threshold with NaN
    data[column][z_scores > threshold] = np.nan

    return data

# Replace outliers in the specified column with NaN values
train = replace_outliers_with_nan(train, ['Estimated_Departure_Time',
                                          'Estimated_Arrival_Time',
                                          'Origin_State',
                                          'Destination_State',
                                          'Airline',
                                          'Carrier_Code(IATA)',
                                          'Carrier_ID(DOT)'], 3)


# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

# Concatenate the training and test sets
concatenated = pd.concat([train.drop('Delay', axis=1), test])

# Fit the label encoder on the concatenated set
for i in qual_col:
    le = LabelEncoder()
    le = le.fit(concatenated[i])
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

# Replace missing values with the mean of each column
for col in NaN:
    concatenated[col] = pd.to_numeric(concatenated[col], errors='coerce')
    mean = concatenated[col].mean() # calculate mean of the column
    train[col] = train[col].fillna(mean) # fill missing values with mean
    test[col] = test[col].fillna(mean)

print('Done.')

# Quantify qualitative variables

# Remove unlabeled data
#train = train.dropna()

# column4 = {}
# for i, column in enumerate(sample_submission.columns):
#     column4[column] = i

# def to_number(x, dic):
#     return dic[x]

# train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
# print('Done.')

train_x = train.drop(columns=['ID', 'Delay'])
train_y = train['Delay']
test_x = test.drop(columns=['ID'])

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Model and hyperparameter tuning using GridSearchCV
model = XGBClassifier(random_state=42)

param_grid = {
    'learning_rate': [0.8, 0.5],
    'max_depth': [2, 6],
    'n_estimators': [100, 900],
}

grid = GridSearchCV(model,
                    param_grid,
                    cv=cv,
                    scoring='accuarcy',
                    n_jobs=-1,
                    verbose=1)

grid.fit(train_x, train_y)

best_model = grid.best_estimator_

# Model evaluation
val_y_pred = np.round(best_model.predict(val_x))

acc = accuracy_score(val_y, val_y_pred)
f1 = f1_score(val_y, val_y_pred, average='weighted')
pre = precision_score(val_y, val_y_pred, average='weighted')
recall = recall_score(val_y, val_y_pred, average='weighted')

print('Accuracy_score:',acc)
print('F1 Score:f1',f1)

y_pred = np.round(best_model.predict_proba(test_x))
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('c:/study/_save/dacon_airplane/d23submission.csv')