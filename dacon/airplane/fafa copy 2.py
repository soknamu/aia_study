import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from xgboost import XGBClassifier
import time
import optuna
from lightgbm import LGBMClassifier
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
NaN_columns = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']
qual_cols = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

# train['Delay'] = train['Delay'].map({'Delayed': 0, 'Not_Delayed': 1})

# Concatenate the training and test sets
concatenated = pd.concat([train.drop('Delay', axis=1), test])

# Fit the label encoder on the concatenated set
for col in qual_cols:
    le = LabelEncoder()
    le.fit(concatenated[col].astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

for col in NaN_columns:
    concatenated[col] = pd.to_numeric(concatenated[col], errors='coerce')
    mean = concatenated[col].mean()
    train[col] = train[col].fillna(mean)
    test[col] = test[col].fillna(mean)

print(train)

print('Done.')

# Quantify qualitative variables

# Remove unlabeled data
train = train['Delay'].dropna()

column4 = {}
for i, column in enumerate(sample_submission.columns):
    column4[column] = i

def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay',
                              'Delay_num',])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = RobustScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

for i in range(10000):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

    def objective(trial):
                alpha = trial.suggest_loguniform('alpha', 0.0000001, 0.1)
                n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 1, 80)
                optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

                model = LGBMClassifier(
                    alpha=alpha,
                    n_restarts_optimizer=n_restarts_optimizer,
                    optimizer=optimizer,
                )
                
                model.fit(train_x, train_y)
                
                print('light result : ', model.score(val_y, val_y_pred))
                
                y_pred = np.round(model.predict(test_x))


                # Model evaluation
                val_y_pred = np.round(model.predict(val_x))

                score = accuracy_score(val_y, val_y_pred)
                f1 = f1_score(val_y, val_y_pred, average='weighted')
                print('Accuracy_score:',score)
                print('F1 Score:f1',f1)

                y_pred = model.predict_proba(test_x)
                submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
                submission.to_csv('c:/study/_save/dacon_airplane/29submission.csv')