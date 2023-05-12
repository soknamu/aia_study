import random
import os
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from catboost import CatBoostClassifier, Pool
from collections import defaultdict

# 시드고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# baseline을 참고했으며, parquet으로 저장하고 실행하면 csv보다 매우 빠르게 데이터 처리가 가능합니다 :)

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')
    
# Load data
train = pd.read_csv('c:/study/_data/dacon_airplane/train.csv')
test = pd.read_csv('c:/study/_data/dacon_airplane/test.csv')
sample_submission = pd.read_csv('c:/study/_data/dacon_airplane/sample_submission.csv', index_col=0)

# 1. Month, Days 처리
def to_days(x):
    month_to_days = {1:0, 2:31, 3:60, 4:91, 5:121, 6:152, 7:182, 8:213, 9:244, 10:274, 11:305, 12:335}
    return month_to_days[x]

train.loc[:, 'Day'] = train['Month'].apply(lambda x: to_days(x))
train['Day'] = train['Day'] + train['Day_of_Month']

test.loc[:, 'Day'] = test['Month'].apply(lambda x: to_days(x))
test['Day'] = test['Day'] + test['Day_of_Month']

train = train.astype({'Day':object})
test = test.astype({'Day':object})

print("Day Done.")

# 2. Carrier_ID(DOT) 처리 with Airline
cond1 = train['Carrier_ID(DOT)'].isnull()
cond2 = ~train['Airline'].isnull()
print("Carrier_ID(DOT) 복구 가능한 데이터의 개수 :", len(train.loc[cond1 & cond2, :]))

# airline to carrier id, dictinary 만들기
# 모두 데이터가 존재하는 열에서 Dict[Airline] = carrier_ID(DOT) 가 되도록 dictionary 생성
airline_to_cid = {}
for _, row in train[(~train['Carrier_ID(DOT)'].isnull() & ~train['Airline'].isnull())].iterrows():
    airline_to_cid[row['Airline']] = row['Carrier_ID(DOT)']

# 복구하기
def to_cid(x):
    return airline_to_cid[x]

cond1 = train['Carrier_ID(DOT)'].isnull()
cond2 = ~train['Airline'].isnull()
train.loc[cond1&cond2, 'Carrier_ID(DOT)'] = train.loc[cond1&cond2, 'Airline'].apply(lambda x: to_cid(x))

# 복구 안 된 row 빼기
train = train.dropna(subset=['Carrier_ID(DOT)'], how='any', axis=0)

# (Test Data Only)
# Airline, Carrier_Code 둘 다 없으면 최빈 값으로 대체
NaN_col = ['Carrier_ID(DOT)']
cond1 = test['Airline'].isnull()
cond2 = test['Carrier_ID(DOT)'].isnull()

for col in NaN_col:
    mode = test[col].mode()[0]
    test.loc[cond1&cond2, col] = mode

# 나머진 Airline에서 대체
cond1 = test['Carrier_ID(DOT)'].isnull()
cond2 = ~test['Airline'].isnull()
test.loc[cond1&cond2, 'Carrier_ID(DOT)'] = test.loc[cond1&cond2, 'Airline'].apply(lambda x: to_cid(x))

print("Cid Done.")

col_drop = ['Month', 'Day_of_Month', 'Cancelled', 'Diverted', 'Origin_Airport', 'Destination_Airport', 'Carrier_Code(IATA)', 'Airline', 'Origin_State', 'Destination_State']
train = train.drop(col_drop, axis=1)
test = test.drop(col_drop, axis=1)
print("Drop Done.")

def to_minutes(x):
    x = int(x)
    x = str(x)
    if len(x) > 2:
        hours, mins = int(x[:-2]), int(x[-2:])
    else:
        hours, mins = 0, int(x[-2:])
    return hours*60+mins

estimated_times = ['Estimated_Departure_Time', 'Estimated_Arrival_Time']

for ET in estimated_times:
    cond = ~train[ET].isnull()
    train.loc[cond, ET] = train.loc[cond, ET].apply(lambda x: to_minutes(x))
    cond2 = ~test[ET].isnull()
    test.loc[cond2, ET] = test.loc[cond2, ET].apply(lambda x: to_minutes(x))
    
train = train.dropna(subset=['Estimated_Arrival_Time', 'Estimated_Departure_Time'], how ='all', axis=0)

time_flying = defaultdict(int)
time_number = defaultdict(int)

cond_arr2 = ~train['Estimated_Arrival_Time'].isnull()
cond_dep2 = ~train['Estimated_Departure_Time'].isnull()

for _, row in train.loc[cond_arr2 & cond_dep2, :].iterrows():
    OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
    time_flying[(OAID,DAID)] += (row['Estimated_Arrival_Time'] - row['Estimated_Departure_Time'])%1440 # 하루 최대는 1440분
    time_number[(OAID,DAID)] += 1
    
    
for key in time_flying.keys():
    time_flying[key] /= time_number[key]
    
# (Test Data Only)
# 둘 다 없으면 최빈값으로 대체
cond_1 = test['Estimated_Departure_Time'].isnull()
cond_2 = test['Estimated_Arrival_Time'].isnull()

mode = test['Estimated_Departure_Time'].mode()[0]
mode2 = test['Estimated_Arrival_Time'].mode()[0]
test.loc[cond_1&cond_2, ['Estimated_Departure_Time', 'Estimated_Arrival_Time']] = mode, mode2


# Departure만 없을 때,
for index, row in test.loc[test['Estimated_Departure_Time'].isnull(),].iterrows():
    OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
    test.loc[index,'Estimated_Departure_Time'] = \
        (test.loc[index]['Estimated_Arrival_Time'] - time_flying[(OAID, DAID)])%1440
    

# Arrival만 없을 때,
for index, row in test.loc[test['Estimated_Arrival_Time'].isnull(),].iterrows():
    OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
    test.loc[index,'Estimated_Arrival_Time'] = \
        (test.loc[index]['Estimated_Departure_Time'] + time_flying[(OAID, DAID)])%1440

    
# 모두 int로 바꾼다.
estimated_times = ['Estimated_Departure_Time', 'Estimated_Arrival_Time']
train = train.astype({'Estimated_Departure_Time':int, 'Estimated_Arrival_Time':int})
test = test.astype({'Estimated_Departure_Time':int, 'Estimated_Arrival_Time':int})
for ET in estimated_times:
    train.loc[train[ET] == 1440, ET] = 0
    test.loc[test[ET] == 1440, ET] = 0


print("EDT, EAT Done.")

# EDT, EAT 48개의 bins에 담으면 된다. 1440(60*24) 계니까, 48씩 끊어서 하면 될 듯
estimate_times = ['Estimated_Departure_Time', 'Estimated_Arrival_Time']
names = {'Estimated_Departure_Time':'EDT', 'Estimated_Arrival_Time':'EAT'}
for ET in estimated_times:
    for i in range(48):
        train.loc[train[ET].between(i*30, (i+1)*30, 'left'), names[ET]] = i
        test.loc[test[ET].between(i*30, (i+1)*30, 'left'), names[ET]] = i

train = train.astype({'EDT':int, 'EAT':int})
test = test.astype({'EDT':int, 'EAT':int})

train = train.drop(['Estimated_Departure_Time', 'Estimated_Arrival_Time'], axis=1)
test = test.drop(['Estimated_Departure_Time', 'Estimated_Arrival_Time'], axis=1)

print("EDT, EAT Done.")

for i in range(51):
    train.loc[train['Distance'].between(i*100, (i+1)*100, 'left'), 'Distance'] = i
    test.loc[test['Distance'].between(i*100, (i+1)*100, 'left'), 'Distance'] = i

train = train.astype({'Distance':int})
test = test.astype({'Distance':int})

print("distance Done.")

train = train.astype({'Carrier_ID(DOT)':int})
test = test.astype({'Carrier_ID(DOT)':int})

train = train.astype({'EDT':object, 'EAT':object, 'Distance':object, 'Origin_Airport_ID':object, \
                     'Destination_Airport_ID':object, 'Carrier_ID(DOT)':object})
test = test.astype({'EDT':object, 'EAT':object, 'Distance':object, 'Origin_Airport_ID':object, \
                     'Destination_Airport_ID':object, 'Carrier_ID(DOT)':object})

print("CID Done.")

train = train.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
print(train.shape)
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

print('Training Prepared.')

counts = list(train_y.value_counts())
class_weight = [counts[1]/sum(counts), counts[0]/sum(counts)]
print("weight :", class_weight)

cat_features = [i for i in range(8)]
model = CatBoostClassifier(random_seed=42, cat_features=cat_features, class_weights=class_weight, verbose=0)
model.fit(train_x, train_y)

y_pred = model.predict_proba(test_x)

submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('c:/study/_save/dacon_airplaneX/catboost_self_weight2.csv', index=True)

print("CSV Done.")