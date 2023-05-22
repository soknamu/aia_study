#전이학습. -> 남이쓰던모델 가져와서 쓰겠다.
from tensorflow.keras.applications import VGG16 #레이어가 16개(conv2D 13개, Dense 3개)

model = VGG16()

# model.summary()

print(model.weights)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________