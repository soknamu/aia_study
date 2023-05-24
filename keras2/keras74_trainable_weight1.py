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

# model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 3)                 6  
# weight = numpy=array([[-1.0895486 , -0.27931535,  1.1966325 ]] ->dense weight 초기값. 
# bias = numpy=array([0., 0., 0.] -> bias의 초기값.
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 8
# shape=(3, 2) array([[-0.8245892 ,  0.91295385],
                #    [ 0.8748083 , -0.05577147],
                #    [-0.11264688,  1.0865679 ]] ->dense_1 weight 초기값. 
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 3
# numpy=array([[1.1687833 ],
#              [0.83164036]]

# =================================================================
# Total params: 17
# Trainable params: 17    
# Non-trainable params: 0
# _________________________________________________________________

print(model.weights)
print("==========================================")
print(model.trainable_weights)
print("==========================================")

print(len(model.weights)) #6 -> 
print(len(model.trainable_weights))#6

model.trainable = False # 
print("==========================================")

print(len(model.weights)) #6 -> 
print(len(model.trainable_weights))#0

print("==========================================")
print(model.weights)
print("==========================================")
print(model.trainable_weights) #[] 아무값도 안들어있음.
print("==========================================")
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 3)                 6
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 8
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 3
# =================================================================
# Total params: 17
# Trainable params: 0 -> 위의것과 차이점
# Non-trainable params: 17
# _________________________________________________________________