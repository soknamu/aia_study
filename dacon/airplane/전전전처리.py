import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures,MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, log_loss
from xgboost import XGBClassifier
import time
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
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

# Cross-validation with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param = {
    'learning_rate': (0.01, 0.3),
    'max_depth' : (3, 16),
    'num_leaves': ( 64, 256),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50),
}

def lgb_cv(learning_rate, max_depth, num_leaves, 
           min_child_samples, min_child_weight, subsample, 
           colsample_bytree, reg_lambda, reg_alpha, max_bin):
    
    # LightGBM 모델의 하이퍼파라미터 설정
    params = {
        'boosting_type': 'gbdt',
        # 'objective': 'Classifier',
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_bin': int(round(max_bin)),
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'device': 'gpu'
    }
    
    # LightGBM 모델 학습
    model = LGBMClassifier(**params)
    model.fit(train_x, train_x, 
              verbose=1)

    # Model evaluation
    val_y_pred = model.predict(val_x)

    # acc = accuracy_score(val_y, val_y_pred)
    # f1 = f1_score(val_y, val_y_pred, average='weighted')
    log = log_loss(val_y, val_y_pred)

    return - log, model, test_x

# BayesianOptimization 객체를 사용하여 lgb_cv 함수를 호출합니다.

optimizer = BayesianOptimization(
    f = lgb_cv,
    pbounds = param,
    random_state = 337,
    verbose = 2
)

result = optimizer.maximize()

# 학습된 모델과 테스트 데이터를 반환합니다.
model = result['params']['model']
test_x = result['params']['test_x']

# 모델을 사용하여 예측을 수행합니다.
y_pred = model.predict_proba(test_x)

# 결과를 저장합니다.
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('c:/study/_save/dacon_airplane/0000submission.csv', float_format='%.3f')


#1708
# param_grid = {
#     'learning_rate': [0.0001, 0.05],
#     'max_depth': [4,6],
#     'n_estimators': [600, 1000],
# }


#1737
# param_grid = {
#     'learning_rate': [0.04, 0.01],
#     'max_depth': [2,6],
#     'n_estimators': [600, 1300],
# }