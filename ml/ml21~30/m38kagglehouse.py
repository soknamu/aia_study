from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
# print(train_csv.shape, test_csv.shape)
# print(train_csv.columns, test_csv.columns)

le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
# 1.3 결측지 확인

# x = train_csv.drop(['SalePrice'], axis=1)
# y = train_csv['SalePrice']
# print(train_csv.shape, test_csv.shape) #(1460, 80) (1459, 79)
# print(train_csv.columns, test_csv.columns)

# 1.3 결측지 확인
y = train_csv['SalePrice']
train_csv.drop(['SalePrice'], axis=1, inplace=True) # SalePrice 드롭
#print(train_csv.shape, test_csv.shape)
x = train_csv


# 1.5 x, y 분리

scaler = StandardScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

bogan = [train_csv.fillna(0),]
        #train_csv.fillna(method='ffill'),]
        # train_csv.fillna(method='bfill'),
        # train_csv.fillna(value = 77)]
test_csv = pd.DataFrame(test_csv)
test_csv = train_csv.fillna(0)

for j in bogan :
    x = j

    # 1.6 train, test 분리
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

    # 1.7 Scaler


    # 2. 모델구성
    model = RandomForestRegressor(random_state=123)

    #3. 훈련
    model.fit(x_train,y_train)

    # 4. 평가, 예측
    loss = model.score(x_test, y_test)
    print('loss : ', loss)

    y_predict = np.round(model.predict(x_test))

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_predict)
    print('r2 : ', r2)

# 4.1 내보내기
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
y_submit = model.predict(test_csv)

y_submit = pd.DataFrame(y_submit)
y_submit = np.array(y_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['SalePrice'] = y_submit
submission.to_csv(path_save + 'kaggle_house_' + date + '.csv')