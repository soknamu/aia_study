import pandas as pd


path = 'd:/study_data/_data/dacon_airplane/'
save_path = 'd:/study_data/_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항

#print(train_csv.shape, test_csv.shape)
#print(train_csv.columns, test_csv.columns)
#print(train_csv.info())

#1.3 결측지 확인
#print(train_csv.isnull().sum())

# 1.5 x, y 분리
x = train_csv.drop(['Delay'], axis=1)
y = train_csv['Delay']

print(x.columns) #(1000000, 17)