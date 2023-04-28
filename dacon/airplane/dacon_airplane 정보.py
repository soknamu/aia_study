import pandas as pd


path = 'c:/study/_data/dacon_airplane/'
save_path = 'c:/study/_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.2 확인사항
print(train_csv.shape, test_csv.shape)

print(train_csv.columns, test_csv.columns)

print(train_csv.info())

# Index: 1000000 entries, TRAIN_000000 to TRAIN_999999
# Data columns (total 18 columns):
#  #   Column                    Non-Null Count    Dtype
# ---  ------                    --------------    -----
#  0   Month                     1000000 non-null  int64
#  1   Day_of_Month              1000000 non-null  int64
#  2   Estimated_Departure_Time  890981 non-null   float64
#  3   Estimated_Arrival_Time    890960 non-null   float64
#  4   Cancelled                 1000000 non-null  int64
#  5   Diverted                  1000000 non-null  int64
#  6   Origin_Airport            1000000 non-null  object
#  7   Origin_Airport_ID         1000000 non-null  int64
#  8   Origin_State              890985 non-null   object
#  9   Destination_Airport       1000000 non-null  object
#  10  Destination_Airport_ID    1000000 non-null  int64
#  11  Destination_State         890921 non-null   object
#  12  Distance                  1000000 non-null  float64
#  13  Airline                   891080 non-null   object
#  14  Carrier_Code(IATA)        891010 non-null   object
#  15  Carrier_ID(DOT)           891003 non-null   float64
#  16  Tail_Number               1000000 non-null  object
#  17  Delay                     255001 non-null   object
# dtypes: float64(4), int64(6), object(8)

#1.3 결측지 확인
print(train_csv.isnull().sum())

# Month                            0
# Day_of_Month                     0
# Estimated_Departure_Time    109019
# Estimated_Arrival_Time      109040
# Cancelled                        0
# Diverted                         0
# Origin_Airport                   0
# Origin_Airport_ID                0
# Origin_State                109015
# Destination_Airport              0
# Destination_Airport_ID           0
# Destination_State           109079
# Distance                         0
# Airline                     108920
# Carrier_Code(IATA)          108990
# Carrier_ID(DOT)             108997
# Tail_Number                      0
# Delay                       744999
