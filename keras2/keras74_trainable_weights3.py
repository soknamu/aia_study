import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))

print(model.weights)
########################################
model.trainable = False
# 이기능을 쓰는 이유 :
# 시드가 고정되어도 틀어지는 경우가 발생하기 때문에
# 전이학습, 사전학습
# 남이 만든 모델을 재학습할 필요가 없을 때
#이유 :사전 훈련된 모델을 사용하는 경우, 특정 레이어의 가중치 고정,
#      특정 파라미터만 업데이트
########################################
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')

model.fit(x,y, batch_size=1, epochs=50)

y_predict = model.predict(x)
print(y_predict)