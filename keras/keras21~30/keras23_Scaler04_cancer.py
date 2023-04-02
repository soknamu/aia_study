import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)       # 판다스 : .describe()
# print(datasets.feature_names)         # 판다스 : .columns()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)     # (569, 30) (569,)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dense(9, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(7, activation='linear'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse',] # 'acc', 'mean_squared_error'
)
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
# print("======================")
# print(y_test[:5])
# print(y_predict[:5])
# print("======================")

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# acc :  0.9035087719298246

# (MinMaxScaler) 
# acc :  0.956140350877193

# (StandardScaler) 
# acc :  0.9385964912280702

# (MaxAbsSclaer) 
# acc :  0.9649122807017544

# (RobustScaler)
# acc :  0.9649122807017544