#세로 개수
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#1. 데이터

path = './_data/wine/'
path_save = './_save/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# train_csv.dropna(inplace=True)
# test_csv.dropna(inplace=True)

#1-1. 라벨인코더

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
train_csv['type'] = le.transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']

print(x.shape,y.shape)

def remove_outliers(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return clean_data

x_clean = remove_outliers(x)
y_clean = remove_outliers(y)

train_csv.dropna(inplace=True)
test_csv.dropna(inplace=True)

#이상치를 지워도 똑같다.

print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 3377, train_size= 0.7, stratify=y
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성

model = RandomForestClassifier(random_state= 3377)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)

# import matplotlib.pyplot as plt

# # create a grouped DataFrame based on the wine quality data
# grouped_df = pd.DataFrame({'quality': y_train, 'features': pd.DataFrame(x_train, columns=x.columns).columns}).groupby('quality').count()

# # plot the grouped data as a bar graph
# grouped_df.plot(kind='bar')

# # set the title and axis labels
# plt.title('Wine Quality Distribution')
# plt.xlabel('Quality')
# plt.ylabel('Count')

# # show the plot
# plt.show()

count_data = train_csv.groupby('quality')['quality'].count()
print(count_data.index)
import matplotlib.pyplot as plt
plt.bar(count_data.index, count_data)
plt.show()








#가로   3  4  5  6  7  8  9
#1. value_counts -> X
#2. np.unique의 return_count X
#3. groupby쓰기 count() 쓰기. 

#plt.bar로 그린다. (quality 컬럼)
#힌트
# 데이터 개수(y축) = 데이터개수.주저리 주저리(groupby)....