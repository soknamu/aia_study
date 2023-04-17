import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
import pandas as pd
warnings.filterwarnings(action = 'ignore')

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape) #(652, 9)
#print(test_csv.shape) #(116, 8)

#print(train_csv.isnull().sum()) # 결측치 x
x = train_csv.drop(['Outcome'],axis =1)
y = train_csv['Outcome']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413)

#2.모델구성
model = RandomForestClassifier()

#3,4. 컴파일, 훈련, 예측
scores = cross_val_score(model, x, y, cv = kfold)
#scores = cross_val_score(model, x, y, cv = 5)
#print(scores) #[0.93333333 0.93333333 0.93333333 1.         0.96666667]

print('ACC :', scores, 
      '\n cross_val_score average : ', round(np.mean(scores),4))

# ACC : [0.79389313 0.70229008 0.7        0.76153846 0.73846154] 
#  cross_val_score average :  0.7392