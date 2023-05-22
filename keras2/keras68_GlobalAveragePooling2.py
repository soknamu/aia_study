#keras32 mnist3

from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 100)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 100)

# print(np.unique(y_train, return_counts=True))
# print(type(x_train))
# print(x_train)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='min', patience=150, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=150, batch_size=128, verbose=1, 
                 validation_split=0.2, callbacks=[es])
end = time.time()

# 걸린 시간 계산
elapsed_time = end - start

# 분과 초로 변환
minutes = elapsed_time // 60
seconds = elapsed_time % 60

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')
# 출력
print("걸린 시간: {}분 {}초".format(int(minutes), int(seconds)))

# Flatten했을 때 epochs = 20
#acc : 0.2904
# 걸린 시간: 0분 20초

# acc : 0.2697 epochs = 50
# 걸린 시간: 0분 46초

# acc : 0.2471 epochs = 150
# 걸린 시간: 1분 31초

# acc : 0.252 epochs = 100, model.add(Dense(50, activation='relu'))추가.
# 걸린 시간: 1분 36초

# acc : 0.3758
# 걸린 시간: 1분 47초

# GlobalAveragePooling2D했을 때 epochs = 20
# acc : 0.2033
# 걸린 시간: 0분 20초

# acc : 0.2776 epochs = 50
# 걸린 시간: 0분 47초

# acc : 0.0689 epochs = 150
# 걸린 시간: 1분 34초

# acc : 0.3258 epochs = 100, model.add(Dense(50, activation='relu'))추가.
# 걸린 시간: 1분 41초

# acc : 0.0851
# 걸린 시간: 0분 25초