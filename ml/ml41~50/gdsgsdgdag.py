from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

# load data
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=369, train_size=0.8, shuffle=True)

# scale data
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train XGBRegressor model
parameters = {'n_estimators': 1000,
              'learning_rate': 0.3,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 0.5,
              'colsample_bytree': 1,
              'colsample_bylevel': 1.0,
              'colsample_bynode': 1,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state': 369}
model = XGBRegressor()
model.set_params(**parameters, early_stopping_rounds=10, eval_metric='rmse')
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=False)

# feature selection using SelectFromModel
for threshold in np.sort(model.feature_importances_):
    selection_model = XGBRegressor()
    selection_model.set_params(**parameters, early_stopping_rounds=10, eval_metric='rmse')
    selection = SelectFromModel(selection_model, threshold=threshold, prefit=False)
    select_x_train = selection.fit_transform(x_train, y_train)
    select_x_test = selection.transform(x_test)
    selection_model.fit(select_x_train, y_train, eval_set=[(select_x_train, y_train), (select_x_test, y_test)], verbose=False)
    selection_y_predict = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, selection_y_predict)
    print("Tresh=%.3f, n=%d, R2: %.2f%%" % (threshold, select_x_train.shape[1], r2 * 100))
    
    
    
# Check label distribution
print(train_csv['quality'].value_counts())

# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Split the data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=850, train_size=0.7, stratify=y)

# Train the model
model = XGBClassifier()
model.set_params(**parameters, early_stopping_rounds=10, eval_metric='merror')

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=True)

