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
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
########################################
model.trainable = False
########################################
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam')

model.fit(x,y, batch_size=1, epochs=50)

y_predict = model.predict(x)
print(y_predict)

# model.trainable = True
# [[1.3275863]
#  [2.168451 ]
#  [3.0093157]
#  [3.8501809]
#  [4.6910458]]

# model.trainable = False
# [[0.18704942]
#  [0.37409884]
#  [0.56114817]
#  [0.7481977 ]
#  [0.9352467 ]]
# -> 가중치가 갱신되지 않는다. 미분이 안되있음. 