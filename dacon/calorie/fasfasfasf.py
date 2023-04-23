import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingRandomSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import datetime
import warnings
warnings.filterwarnings('ignore')

xgb_model = xgb.XGBRegressor()

scaler_list = [MinMaxScaler(), MaxAbsScaler(), StandardScaler(), RobustScaler()]
model_list = [GaussianProcessRegressor(), CatBoostRegressor()]

param_r = [{"kernel":[ C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2)) + WhiteKernel(noise_level=1.2, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.5, (1e-3, 1e3)) * RBF(15, (1e-2, 1e2)) + WhiteKernel(noise_level=1.5, noise_level_bounds=(1e-10, 1e+1)),],
               "n_restarts_optimizer": [5, 9, 12],"alpha": [1e-10, 1e-5],}]

param_d = [{'iterations':[1000,2000,1500],'learning_rate':[0.03,0.05,0.01]},{'depth':[6,5,2],'loss_function':['RMSE'],'task_type':['CPU']}]

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
regressor = all_estimators(type_filter='regressor')

path = 'c:/_study/_data/_dacon_Calories_Burned/'
save_path = 'c:/_study/_save/dacon_Calories_Burned/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(Feet)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
x['Height(Remainder_Inches)'] = 703*x['Weight(lb)']/x['Height(Feet)']**2

test_csv['Height(Feet)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']
test_csv['Height(Remainder_Inches)'] = 703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2

le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

for k in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=k)

    for i in scaler_list:
        scaler = i
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        test_csv = scaler.transform(test_csv)

        for j in range(len(model_list)):
            if j==0:
                param = param_r
            elif j==1:
                param = param_d
            model = RandomizedSearchCV(model_list[j], param, cv=10, verbose=1)
            model.fit(x_train, y_train)

            loss = model.score(x_test, y_test)
            print('loss : ', loss)
            print('test RMSE : ', RMSE(y_test, model.predict(x_test)))
            
            if RMSE(y_test, model.predict(x_test))<0.5:
                submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
                submit_csv['Calories_Burned'] = model.predict(test_csv)
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(save_path + 'Calories' + date + '.csv')
                break