import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)       # 판다스 .describe()
print(datasets.feature_names)       # 판다스 columns

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)     # (150, 4), (150,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))        # y의 라벨값 : [0 1 2]
print(y)

################# 요지점에서 원핫
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)

# 판다스에 겟더미, 사이킷런에 원핫인코더 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    # random_state=333,
    train_size=0.8,
    stratify=y
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(y_train)
print(np.unique(y_train, return_counts=True))


# 2. 모델구성
# model = Sequential()
# model.add(Dense(50, activation='relu', input_dim=4))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))

input1 = Input(shape=(4,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(1, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

# accuracy_score를 사용해서 스코어를 빼세요
"""
from sklearn.metrics import accuracy_score

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=-1)
print(y_predict.shape)

y_test = np.argmax(y_test, axis=-1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)"""

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
# print(y_test.shape)     # (30, 3)

# print(y_pred.shape)     # (30, 3)

# print(y_test[:5])
# print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis=1)      # 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis=-1)      

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy score : ', acc)


# loss :  0.01928967796266079
# acc :  1.0
# accuracy score :  1.0
 
# (MinMaxScaler) 
# loss :  0.0990469828248024
# acc :  0.9666666388511658
# accuracy score :  0.9666666666666667

# (StandardScaler) 
# loss :  0.005187265109270811
# acc :  1.0
# accuracy score :  1.0

# (MaxAbsSclaer) 
# loss :  0.056688953191041946
# acc :  0.9666666388511658
# accuracy score :  0.9666666666666667

# (RobustScaler)
# loss :  0.0014884460251778364
# acc :  1.0
# accuracy score :  1.0