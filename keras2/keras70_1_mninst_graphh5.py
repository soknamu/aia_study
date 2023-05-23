from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(337)

# 1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, 28*28))
x_test = scaler.transform(x_test.reshape(-1, 28*28))
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_train = np.array(pd.get_dummies(y_train, prefix='number'))
y_test = np.array(pd.get_dummies(y_test, prefix='number'))

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(33, 2))
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='min', patience=100, verbose=1, restore_best_weights=True)

# 모델 저장 콜백
model_save = ModelCheckpoint('./_save/keras70로드.h5', save_best_only=True, monitor='acc', mode='max', verbose=1)

history = model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es, model_save])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])

