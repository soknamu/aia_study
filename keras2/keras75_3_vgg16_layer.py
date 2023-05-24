import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3), )

# vgg16.trainable = False # 가중치 동결.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.trainable = False # 모델전체 동결하는거.

# model.summary()

# print(len(model.weights))
# print(len(model.trainable_weights))

# print(model.layers)
# [<keras.engine.functional.Functional object at 0x000001CD0873E0D0>, vgg16
#  <keras.layers.core.flatten.Flatten object at 0x000001CD08745700>,  flatten
#  <keras.layers.core.dense.Dense object at 0x000001CD0878FE80>,      Dense
#  <keras.layers.core.dense.Dense object at 0x000001CD087A4610>]      Dense

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
# print(layers)
# [(<keras.engine.functional.Functional object at 0x000002438525E640>, 'vgg16', False), 
#  (<keras.layers.core.flatten.Flatten object at 0x0000024385263640>, 'flatten', True), 
#  (<keras.layers.core.dense.Dense object at 0x00000243852AFDC0>, 'dense', True), 
#  (<keras.layers.core.dense.Dense object at 0x00000243852C4400>, 'dense_1', True)]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer name', 'Layer Trainable'])
print(results)
#                             Layer Type                            Layer name  Layer Trainable
# 0  <keras.engine.functional.Functional object at 0x000001AD82FFE2E0>  vgg16      False
# 1  <keras.layers.core.flatten.Flatten object at 0x000001AD830038B0>   flatten    True
# 2  <keras.layers.core.dense.Dense object at 0x000001AD8304FF40>       dense      True
# 3  <keras.layers.core.dense.Dense object at 0x000001AD830604F0>       dense_1    True