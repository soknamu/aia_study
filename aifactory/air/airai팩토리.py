
import pandas as pd
from sklearn.svm import OneClassSVM
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
...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Train One-Class SVM model on train data
svm = OneClassSVM(kernel='rbf', nu=0.05) # nu is the fraction of training data to be considered as outliers
svm.fit(train_data)

# Predict anomalies in test data
predictions = svm.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})

# Save predictions to submission file
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
submission.to_csv(save_path +'air_' + date + '.csv', index=False)
