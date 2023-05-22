#keras32 mnist3

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########### 실습 #############
scaler=MinMaxScaler()
x_train = x_train.reshape(-1,1)
# print(x_train.shape)
# print((x_train.shape[0]//(28*28)))
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))


# print(np.unique(y_train, return_counts=True))
# print(type(x_train))
# print(x_train)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(33, 2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu')) #앞의 필터의 개수만큼 노드로 만들어줌 33개의 평균값을내서 하나로 만듬.
model.add(Dense(10, activation='softmax'))
model.summary()

#                         flatten 했을때
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
# _________________________________________________________________
# flatten (Flatten)            (None, 4608)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                46090
# =================================================================
# Total params: 71,082
# Trainable params: 71,082
# Non-trainable params: 0
# _________________________________________________________________

#                    GlobalAveragePooling2D했을때

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 32)                0
# _________________________________________________________________
# dense (Dense)                (None, 10)                330
# =================================================================
# Total params: 25,322
# Trainable params: 25,322
# Non-trainable params: 0  
# _________________________________________________________________

# 실직적으로 연산량이 많은 곳은 flatten한 후에 Dense부분임 
# 원래 flatten은 펴주기만하고, 계산이 없는 곳인데 flatten 실질적으로는 conv2d에서 연산량이 많아져야 좋은데 flatten후에가 많아져서 
# 그걸 방지하고자 GlobalAveragePooling2D사용
'''
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='min', patience=100, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
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

# flatten 했을때
# acc : 0.9877
# 걸린 시간: 0분 47초

#GlobalAveragePooling2D 했을때
#acc : 0.9228
# 걸린 시간: 0분 49초

# acc : 0.9597 GlobalAveragePooling2D
# 걸린 시간: 1분 43초

# acc : 0.9891 Flatten
# 걸린 시간: 1분 49초
'''