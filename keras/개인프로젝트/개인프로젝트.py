import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/private_data/population/'
path_save = './_save/samsung/'

# Read the four CSV files
csv1 = pd.read_csv(path +'file1.csv')
csv2 = pd.read_csv('file2.csv')
csv3 = pd.read_csv('file3.csv')
csv4 = pd.read_csv('file4.csv')

# Concatenate the four CSV files vertically
combined_csv = pd.concat([csv1, csv2, csv3, csv4], axis=0)

# Write the combined data to a new CSV file
combined_csv.to_csv('combined.csv', index=False)