import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import OneClassSVM
# Load train and test data
path='./_data/air/'
save_path= './_save/air/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')
# Combine train and test data
data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
numeric_features = ['air_inflow', 'air_end_temp', 'motor_current', 'motor_rpm','motor_temp','motor_vibe']
categorical_features = ['type']
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

data = preprocessor.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

# Split back into train and test data
train_data_pca = pca_data[:len(train_data), :]
test_data_pca = pca_data[len(train_data):, :]

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
