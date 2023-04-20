import pandas as pd
import datetime
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터 경로 지정
path = './_data/air/'
save_path = './_save/air/'

# 데이터 로드
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

# Select subset of features for LOF model
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size=0.9, random_state=5555)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert pandas dataframe to H2O dataframe
h2o.init()
h2o.remove_all()

train_h2o = h2o.H2OFrame(train_data)
test_h2o = h2o.H2OFrame(test_data)

# Define columns
train_h2o.columns = list(train_data.columns)
test_h2o.columns = list(test_data.columns)

# Define target column
target = 'label'

# Run AutoML
aml = H2OAutoML(max_runtime_secs=600, seed=123)
aml.train(y=target, training_frame=train_h2o)

# Get the best model from AutoML
best_model = aml.leader

# Make predictions on validation set
y_pred_val = best_model.predict(h2o.H2OFrame(X_val))
y_pred_val = y_pred_val.as_data_frame()['predict'].tolist()

# Tuning: Adjust the threshold for predicting anomalies
threshold = 0.5
y_pred_val = [1 if x > threshold else 0 for x in y_pred_val]

# Make predictions on test set
y_pred_test = best_model.predict(test_h2o)
y_pred_test = y_pred_test.as_data_frame()['predict'].tolist()

# Tuning: Adjust the threshold for predicting anomalies
threshold = 0.5
y_pred_test = [1 if x > threshold else 0 for x in y_pred_test]

submission['label'] = pd.DataFrame({'Prediction': y_pred_test})
print(submission.value_counts())

# Save submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + 'air_' + date + '.csv', index=False)
