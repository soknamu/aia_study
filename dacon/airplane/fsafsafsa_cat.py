import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import time
# Load data
train = pd.read_csv('c:/study/_data/dacon_airplane/train.csv')
test = pd.read_csv('c:/study/_data/dacon_airplane/test.csv')
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col=0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
NaN = ['Origin_State', 'Destination_State', 'Airline', 'Estimated_Departure_Time', 'Estimated_Arrival_Time', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)']

for col in NaN:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)

    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train[i])
    train[i] = le.transform(train[i])

    for label in np.unique(test[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test[i] = le.transform(test[i])
print('Done.')

# Remove unlabeled data
train = train.dropna()

column4 = {}
for i, column in enumerate(sample_submission.columns):
    column4[column] = i

def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column4))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters for tuning
    alpha = trial.suggest_loguniform('alpha', 0.0000001, 0.1)
    n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 1, 80)
    optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

    # Train the model with the hyperparameters
    model = CatBoostClassifier(
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                optimizer=optimizer,)
    
    model.fit(train_x, train_y) 

    # Model evaluation
    val_y_pred = np.round(model.predict(val_x))
    log =log_loss(val_y, val_y_pred)
    f1 = f1_score(val_y, val_y_pred, average='weighted')

    # Print evaluation metrics
    print('Log loss:', log)
    print('F1 Score:', f1)



    return log # You can change the return value to any evaluation metric you want to optimize

# Run 100 trials of the objective function with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Create the CatBoostClassifier model with the best hyperparameters
best_params = study.best_params
model = CatBoostClassifier(
            alpha=best_params['alpha'],
            n_restarts_optimizer=best_params['n_restarts_optimizer'],
            optimizer=best_params['optimizer'])
model.fit(train_x, train_y)

# Generate predictions on the test set
test_x = test.drop(columns='ID')
# Save predictions to a file
y_pred = model.predict_proba(test_x)
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('c:/study/_save/dacon_airplane/43742submission.csv')
