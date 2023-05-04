import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier
import datetime
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Load data
train = pd.read_csv('c:/study/_data/dacon_airplane/train.csv')
test = pd.read_csv('c:/study/_data/dacon_airplane/test.csv')
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col=0)

#print(train)
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

pf = PolynomialFeatures(degree=2)
train_x = pf.fit_transform(train_x)

# Split the training dataset into a training set and a validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler1 = StandardScaler()
scaler2 = StandardScaler()
train_x = scaler1.fit_transform(train_x)
val_x = scaler1.transform(val_x)
test_x = scaler2.fit_transform(test_x)

pf = PolynomialFeatures(degree=2)
train_x = pf.fit_transform(train_x)
test = pf.transform(test)
# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=337)

# Model and hyperparameter tuning using GridSearchCV
path_save ='c:/study/_save/dacon_airplane/'

def objective(x_train, y_train, x_test, y_test, path_save):

    min_logloss = float('inf')

    def train_catboost(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 400, 700),
            'depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.02),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 1),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 0.1)                
            }
        model = CatBoostClassifier(**param, verbose=1,
                                task_type='GPU')
        valid_cv = KFold(n_splits=5, shuffle=True)
        logloss_scores = []
        for train_idx, valid_idx in valid_cv.split(x_train):
            train_x, train_y = x_train[train_idx], y_train.iloc[train_idx]
            valid_x, valid_y = x_train[valid_idx], y_train.iloc[valid_idx]
            # 모델 학습
            model.fit(train_x, train_y, eval_set=(valid_x, valid_y))
            val_y_pred = model.predict(valid_x)
            logloss = log_loss(valid_y, val_y_pred)
            logloss_scores.append(logloss)
        avg_logloss = np.mean(logloss_scores)
        if avg_logloss < min_logloss:
            min_logloss = avg_logloss
            date = datetime.datetime.now().strftime('%m%d_%H%M%S')
            y_pred = np.round(model.predict_proba(test), 3)
            submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
            submission.to_csv(path_save + date + str(round(avg_logloss, 3)) + '.csv', index=True)
        return avg_logloss

    opt = optuna.create_study(direction='minimize')
    opt.optimize(lambda trial: train_catboost(trial), n_trials=100)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
            
            # param = {
            #     'iterations': trial.suggest_int('iterations', 400, 700),
            #     'depth': trial.suggest_int('max_depth', 4, 8),
            #     'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.02),
            #     'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 0, 1),
            #     'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 1),
            #     'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            #     'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            #     'border_count': trial.suggest_int('border_count', 64, 128),
                
            #     }