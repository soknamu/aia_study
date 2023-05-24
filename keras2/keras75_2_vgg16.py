import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3), )

vgg16.trainable = False # 가중치 동결.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# Trainable : True  30, 30
# Trainable : False 30, 0
# vgg16.trainable = False 30, 4

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688

#  flatten (Flatten)           (None, 512)               0

#  dense (Dense)               (None, 100)               51300

#  dense_1 (Dense)             (None, 10)                1010

# =================================================================
# Total params: 14,766,998
# Trainable params: 14,766,998
# Non-trainable params: 0
# _________________________________________________________________
