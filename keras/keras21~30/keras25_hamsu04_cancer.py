from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=30))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(30,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(9, activation='linear')(dense1)
dense3 = Dense(8, activation='linear')(dense2)
dense4 = Dense(7, activation='linear')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)
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

# hamsu
# result :  [0.3187672793865204, 0.9707602262496948, 0.024951957166194916]
# acc :  0.9707602339181286