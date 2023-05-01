import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


path = './_data/spark/TRAIN/'
path_sub = './_save/spark/'
path_save = './_save/spark/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path_save + 'test_input.csv', index_col=0)
submission = pd.read_csv(path_sub+'answer_sample.csv')


'''
48개 아니면 72개


'''