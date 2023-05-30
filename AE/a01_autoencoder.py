# 자동제거, x를 x로 훈련시킨다. 준지도 학습.

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
decoded = Dense(784, activation='tanh')(encoded)
autoencoder = Model(input_img, decoded)

autoencoder.summary()
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 784)]             0

#  dense (Dense)               (None, 64)                50240

#  dense_1 (Dense)             (None, 784)               50960

# =================================================================
# Total params: 101,200
# Trainable params: 101,200
# Non-trainable params: 0
# _________________________________________________________________

#오토인코더의 고질적인 문제 : 사진이 뿌얘질수도 있음. -> 압축 -> 풀기를 하기때문에.
#학습자체가 문제있는 사진, 문제없는 사진 이렇게 학습.

# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train,x_train, epochs =30, batch_size= 128,
                validation_split=0.2)