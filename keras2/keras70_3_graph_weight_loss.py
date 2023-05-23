from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential,load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import tensorflow as tf
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

# 2. 모델구성
model = Sequential()

# history 객체 로드
import pickle

with open('./_save/keras70_1_mnist_grape.pkl', 'rb') as f:
    hist = pickle.load(f)

# 가중치 불러오기
weights = np.load('./_save/keras70_1_mnist_grape_weights.npy', allow_pickle=True)
model.set_weights(weights)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
plt.plot(hist['loss'], marker='.', c='red', label='loss')
plt.plot(hist['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.show()


