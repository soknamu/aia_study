from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1, name='hidden1'))
model.add(Dense(2, name='hidden2'))
model.add(Dense(1, name='outputs'))


# #1.전체동결
# model.trainable = False

#2. 전체동결
# for layer in model.layers:
#     layer.trainable=False
#1번과 동일하게 전체동결됨.

#3. 
# print(model.layers[0]) #<keras.layers.core.dense.Dense object at 0x0000026C13208F40>
# model.layers[0].trainable = False    #hidden1
model.layers[1].trainable = False   #hidden2
# model.layers[2].trainable = False   #outputs

model.summary()

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer name', 'Layer Trainable'])
print(results)

#                          Layer Type                             Layer name  Layer Trainable
# 0  <keras.layers.core.dense.Dense object at 0x000001B4C0B58F40>  hidden1    False
# 1  <keras.layers.core.dense.Dense object at 0x000001B48FD2D280>  hidden2    False
# 2  <keras.layers.core.dense.Dense object at 0x000001B48FD7F790>  outputs    False