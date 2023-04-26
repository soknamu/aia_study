import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,\
    load_digits,fetch_covtype
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from tensorflow.keras.datasets import mnist

#1. 데이터
# x,y = load_iris(return_X_y=True) # (150,4) -> (150,2)
# # x,y = load_digits(return_X_y=True)
# # x,y = load_breast_cancer(return_X_y=True)
# # x,y = load_wine(return_X_y=True)
# # x,y = fetch_covtype(return_X_y=True)

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True),
             fetch_covtype(return_X_y=True)]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'wine :',
                  'covtype :']

data = []
data_name = []

#1. 데이터
for index, value in enumerate(data_list):

    x, y = value 
    lda = LinearDiscriminantAnalysis() 
    x_lda = lda.fit_transform(x,y)

    lda_EVR = lda.explained_variance_ratio_

    cumsum = np.cumsum(lda_EVR)

    data.append(x_lda)
    data_name.append(data_name_list[index])

for i, name in enumerate(data_name_list):
    print(f"{name} shape: {data[i].shape}")
    print(f"{name} cumsum: {cumsum}")

# for i in enumerate(len(data_name_list)):
#     print(f"{data_name_list[i]}, 'shape'{data[i]:.4f}")
#     print(f"{data_name_list[i]}, 'cumsum'{data_name[i]:.4f}")


# iris :  shape: (150, 2)
# iris :  cumsum: [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]        
# breast_cancer : shape: (569, 1)
# breast_cancer : cumsum: [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# digits : shape: (1797, 9)
# digits : cumsum: [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# wine : shape: (178, 2)
# wine : cumsum: [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# covtype : shape: (581012, 6)
# covtype : cumsum: [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
