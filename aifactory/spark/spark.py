import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


path_train = './_data/spark/TRAIN/'

path = './_save/spark/'

train_csv = pd.read_csv(path_train + 'train_all.csv', index_col=0)
test_csv = pd.read_csv(path + 'train.csv', index_col=0)