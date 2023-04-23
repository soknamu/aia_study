import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MaxAbsScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna
import datetime
import warnings
warnings.filterwarnings('ignore')

poly = PolynomialFeatures(degree=2, include_bias=False)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

path = './_data/calorie/'
path_save = './_save/calorie/'
path_save_min = './_save/calorie/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0).drop(['Weight_Status'], axis=1)
#train_csv = pd.read_csv(path + 'train.csv', index_col=0)
#test_csv = pd.read_csv(path + 'test.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0).drop(['Weight_Status'], axis=1)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x['Height(inch)'] = 12*x['Height(Feet)']+x['Height(Remainder_Inches)']
test_csv['Height(inch)'] = 12*test_csv['Height(Feet)']+test_csv['Height(Remainder_Inches)']

x['BMI'] = (703*x['Weight(lb)']/x['Height(Feet)']**2)
test_csv['BMI'] = (703*test_csv['Weight(lb)']/test_csv['Height(Feet)']**2)

le = LabelEncoder()
#le2 = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
#x['Weight_Status'] = le2.fit_transform(x['Weight_Status'])
test_csv['Gender'] = le.transform(test_csv['Gender'])
#test_csv['Weight_Status'] = le2.transform(test_csv['Weight_Status'])

x = x.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)
test_csv = test_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1)

# # 다항식 변환
# x = poly.fit_transform(x)
# test_csv = poly.transform(test_csv)

scaler = MaxAbsScaler()
x = pd.DataFrame(scaler.fit_transform(x))
test_csv = pd.DataFrame(scaler.transform(test_csv))

for i in range(10000):
    kf = KFold(n_splits=8, shuffle=True, random_state=i)

    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 0.0000001, 0.1)
            n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 1, 60)
            optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', 'Powell', 'CG'])

            model = GaussianProcessRegressor(
                alpha=alpha,
                n_restarts_optimizer=n_restarts_optimizer,
                optimizer=optimizer,
            )
            
            model.fit(x_train, y_train)
            
            print('GPR result : ', model.score(x_test, y_test))
            
            y_pred = np.round(model.predict(x_test))
            rmse = RMSE(y_test, y_pred)
            print('GPR RMSE : ', rmse)
            if rmse < 0.21:
                submit_csv['Calories_Burned'] = np.round(model.predict(test_csv))
                date = datetime.datetime.now()
                date = date.strftime('%m%d_%H%M%S')
                submit_csv.to_csv(path_save + date + str(round(rmse, 5)) + '.csv')
            return rmse
        opt = optuna.create_study(direction='minimize')
        opt.optimize(objective, n_trials=20)
        print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)
        
#가우시안
# degree: PolynomialFeatures의 다항식 차수
# alpha: GaussianProcessRegressor의 정규화 하이퍼파라미터
# n_restarts_optimizer: optimizer가 수렴하지 않을 경우 다시 시작하는 횟수
# optimizer: GaussianProcessRegressor의 optimizer 선택