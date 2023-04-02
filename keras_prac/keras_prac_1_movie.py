import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/movie/'
path_save = './_save/movie/'

train_csv = pd.read_csv(path + 'movies_train.csv', index_col=0)
test_csv = pd.read_csv(path + 'movies_test.csv', index_col=0)

# 1.2 확인사항 5가지
print(train_csv.shape, test_csv.shape)      # (600, 11), (243, 10)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))SDSD