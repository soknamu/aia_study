import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

#model = VGG16() include_top = True, input_shape = (224, 224, 3) 증폭을 하면 낭비임. 그리고 output도 안맞아서 안됨.
model = VGG16(weights='imagenet', include_top=False, #True하면 에러뜸. -> 
              input_shape=(32,32,3), )
model.summary()

print(len(model.weights)) #32 16*2 가중치 16개 + bias 16개 -> 26개
print(len(model.trainable_weights)) #32 -> 26개

######################## include_top = True ##########################
#1. FC layer 원래꺼 쓴다.
#2. input_shape(224,224,3) 고정값, - 바꿀 수 없다.

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ........
#  flatten (Flatten)           (None, 25088)             0
#  fc1 (Dense)                 (None, 4096)              102764544
#  fc2 (Dense)                 (None, 4096)              16781312
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
####################################################################

######################## include_top = False ##########################
#1. FC layer 원래꺼 삭제 -> 내가 쓴 값으로 적용됨.
#2. input_shape(32,32,3) - 바꿀 수 있다. -> 맘대로 가능.

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0
#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# ........
# flatten 하단부분(폴리커넥티드 레이어부분) 사라짐.
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
####################################################################