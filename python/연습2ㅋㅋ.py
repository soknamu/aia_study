import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Load data
path = './_data/air/'
save_path = './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Select subset of features
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and validation data
X = train_data[features]
X_train, X_val = train_test_split(X, train_size=0.9, random_state=5555)

# Normalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
test_data_norm = scaler.transform(test_data[features])

# Initialize models
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=5555)
svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')

# Fit models to training data
lof.fit(X_train_norm)
iforest.fit(X_train_norm)
svm.fit(X_train_norm)

# Predict anomalies on validation data
y_pred_lof_val = lof.predict(X_val_norm)
y_pred_iforest_val = iforest.predict(X_val_norm)
y_pred_svm_val = svm.predict(X_val_norm)

# Combine predictions using voting
y_pred_val = np.sum([y_pred_lof_val, y_pred_iforest_val, y_pred_svm_val], axis=0)
y_pred_val = np.where(y_pred_val < 0, -1, 1)

# Predict anomalies on test data
y_pred_lof_test = lof.predict(test_data_norm)
y_pred_iforest_test = iforest.predict(test_data_norm)
y_pred_svm_test = svm.predict(test_data_norm)

# Combine predictions using voting
y_pred_test = np.sum([y_pred_lof_test, y_pred_iforest_test, y_pred_svm_test], axis=0)
y_pred_test = np.where(y_pred_test < 0, -1, 1)

# Save submission file
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test]
submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
date = datetime.datetime.now().strftime("%m%d_%H%M%S")
submission.to_csv(save_path + 'air_' + date + '.csv', index=False)
