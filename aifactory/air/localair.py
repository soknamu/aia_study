import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)

# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data

def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Scale the numerical features
scaler = StandardScaler()
num_features = ['air_inflow', 'air_end_temp', 'motor_current', 'motor_rpm','motor_temp','motor_vibe']
data[num_features] = scaler.fit_transform(data[num_features])
train_data[num_features] = scaler.transform(train_data[num_features])
test_data[num_features] = scaler.transform(test_data[num_features])

# Train Local Outlier Factor model on train data
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05) # n_neighbors is the number of neighbors to consider for LOF, and contamination is the expected percentage of outliers in the dataset
lof.fit(train_data)

# Predict anomalies in test data
predictions = lof.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})

# Save predictions to submission file
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
submission.to_csv(save_path +'air_' + date + '.csv', index=False)