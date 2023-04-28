import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,StandardScaler,RobustScaler

path = 'c:/study/_data/dacon_airplane/'
save_path = 'c:/study/_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col=0)

#1-1. 라벨인코더

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['Origin_Airport',
                 'Origin_State',
                 'Destination_Airport',
                 'Destination_State',
                 'Airline',
                 ])

train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# 1-1 x, y 분리
x = train_csv.drop(['Delay'], axis=1)
y = train_csv['Delay']

#print(x.shape, y.shape) #(1000000, 17) (1000000,)

#1-1 이상치 제거

def remove_outliers(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return clean_data

x_clean = remove_outliers(x)
y_clean = remove_outliers(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=369, train_size=0.8 
)

#1-2 스케일러
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


