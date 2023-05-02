import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier
import datetime
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
path_save = 'c:/study/_save/dacon_airplane/'
scaler_list = [
            #    MinMaxScaler(),
            #    MaxAbsScaler(), 
            #    StandardScaler(), 
               RobustScaler(),
               ]
model_list = [CatBoostClassifier()]

cat = CatBoostClassifier()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Fixed Seed

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('c:/study/_data/dacon_airplane/train.csv', 'train')
csv_to_parquet('c:/study/_data/dacon_airplane/test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col = 0)

# Replace variables with missing values except for the label (Delay) with the most frequent values of the training data
# 컬럼의 누락된 값은 훈련 데이터에서 해당 컬럼의 최빈값으로 대체됩니다.
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

# Quantify qualitative variables
# 정성적 변수는 LabelEncoder를 사용하여 숫자로 인코딩됩니다.
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
# 훈련 세트에서 레이블이 지정되지 않은 데이터가 제거되고 숫자 레이블 열이 추가됩니다.
train = train.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test = test.drop(columns=['ID'])


# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=337)

# Model and hyperparameter tuning using GridSearchCV

for k in range(10):
    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=337, stratify=train_y)

    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test = scaler.transform(test)

        def objective(trial, x_train, y_train, x_test, y_test):
            param = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.00000001, 0.01),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'num_boost_round': 1000000,
                'early_stopping_round': 1000,
                'device': 'gpu'
            }

            model = LGBMClassifier(n_jobs=-1,**param)

            valid_cv = KFold(n_splits = 5,
                shuffle = True)
            # for train_idx, valid_idx in valid_cv.split(x_train):
            
            #     train_x , test_x = x_train.iloc[train_idx], y_train.iloc[train_idx]
            #     valid_x , valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]
            
            # 모델 학습
           
            model.fit(x_train, y_train,
                      eval_set=[(x_train,y_train),
                                (x_test,y_test)],
                      verbose=1)
            val_y_pred = model.predict(x_test)
            f1 = f1_score(y_test, val_y_pred, average='weighted')
            precision = precision_score(y_test, val_y_pred, average='weighted')
            recall = recall_score(y_test, val_y_pred, average='weighted')
            logloss = log_loss(y_test, val_y_pred)
            print(f'F1 Score: {f1}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'logloss : {logloss}')
            y_pred = np.round(model.predict_proba(test),3)
            # y_pred = np.round(y_pred, 5)

            submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
            date = datetime.datetime.now()
            date = date.strftime('%m%d_%H%M%S')
            submission.to_csv(path_save + date +str(round(logloss, 3))+'.csv', index=True)

        opt = optuna.create_study(direction='minimize')
        opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test), n_trials=10000)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)