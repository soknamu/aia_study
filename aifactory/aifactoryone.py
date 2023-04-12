import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Define preprocessing steps for different column types
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['temperature', 'humidity', 'windspeed']),
        ('cat', categorical_transformer, ['type'])
    ])
# Fit preprocessing on training data and transform all data
data = preprocessor.fit_transform(data)

# Split data back into train and test sets
X_train = data[:len(train_data)]
X_test = data[len(train_data):]

# Train One-Class SVM model on train data
from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel='linear', nu=0.045)
model.fit(X_train)

# Predict anomalies in test data
predictions = model.predict(X_test)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(save_path + 'air_ocs.csv', index=False)
