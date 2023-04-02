import numpy as np
from tensorflow.python.keras import Sequential #models안적어도 가능
from tensorflow.python.keras.layers import Dense

#1. 데이터

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
#0 칼럼을 삭제하기위해서 sklearn이랑 get_dummies를 사용.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 10  -> None 이라고 뜨는이유:행의 갯수는 상관없어서
# _________________________________________________________________ ->바이어스(b)까지 포함해서 10
# dense_1 (Dense)              (None, 4)                 24          -> sumary로 어느정도 연산이 되는지 파악.
# _________________________________________________________________  -> model.add(Dense(3)) * model.add(Dense(3)) = 9
# dense_2 (Dense)              (None, 3)                 15
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 8
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 3
# =================================================================
# Total params: 60
# Trainable params: 60
# Non-trainable params: 0
# _________________________________________________________________

