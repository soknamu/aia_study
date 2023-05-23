#keras32 mnist3

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential,load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########### 실습 #############
scaler=MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

loaded_model = tf.keras.models.load_model('./_save/keras70로드.h5')
# history 파일 로드
history = {}
with h5py.File('./_save/keras70로드.h5', 'r') as hf:
    for key in hf.keys():
        history[key] = hf[key][()]

import matplotlib.pyplot as plt

# Loss 그래프
plt.subplot(2, 1, 1)
plt.plot(history['loss'], marker='.', c='red', label='loss')
plt.plot(history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(2, 1, 2)
plt.plot(history['acc'], marker='.', c='red', label='acc')
plt.plot(history['val_acc'], marker='.', c='blue', label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# plt.show()
# plt.figure(figsize=(9,5))
# plt.plot(hist['loss'], marker='.', c='red', label='loss')
# plt.plot(hist['val_loss'], marker='.', c='blue', label='val_loss')
# plt.plot(hist['acc'], marker='.', c='red', label='acc')
# plt.plot(hist['val_acc'], marker='.', c='blue', label='val_acc')
# plt.title('Training History')
# plt.xlabel('Epochs')
# plt.ylabel('Metrics')
# plt.legend()
# plt.show()