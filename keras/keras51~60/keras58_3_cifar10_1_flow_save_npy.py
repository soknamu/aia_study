from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
import random
import pandas as pd
save_path = 'd:/study_data/_save/cifar10/'

#0. seed initialization

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

(x_train, y_train), (x_test,y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
    vertical_flip=True, #뒤집기
    width_shift_range= 0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode= 'nearest')

train_datagen2 = ImageDataGenerator(
    rescale= 1./1,
)


augment_size = 25000

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy() # x_augmented 에 4만개가 들어감. copy를 통해서 x_train데이터가 덮어씌어지지 않음.
y_augmented = y_train[randidx].copy()

# print(x_augmented)
# print(x_augmented.shape, y_augmented.shape) #(40000,28,28) , (40000,)


#차원 바꾸기(3-> 4)

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(x_test.shape[0],
#                        x_test.shape[1], 
#                        x_test.shape[2], 3)
# #x_test = x_test.shape(10000, 28, 28, 1) 와 동일.

# x_augmented = x_augmented.reshape(x_augmented.shape[0],
#                                   x_augmented.shape[1],
#                                   x_augmented.shape[2], 1)


#변환

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False
).next()[0]

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
x_test = x_test/255.

print(x_train.shape, y_train.shape)

#print(np.max(x_train), np.min(x_train)) #1.0 0.0

# y_train = np.array(pd.get_dummies(y_train))
# y_test = np.array(pd.get_dummies(y_test))
# 이렇게 하면은 test랑 train이랑 순서를 바뀌면 제대로 원핫이 안되기때문에.
# 문자열이나 순서가 제대로 안되있으면 제대로 원핫을 못해서 위에 것 처럼 더해줌.

y_onehot = pd.get_dummies(np.concatenate((y_train,y_test)).reshape(-1), prefix='number')

#reshape(-1) (75000, 1)을 (75000,) 으로 만들어줘야하기 때문에

y_train = y_onehot[:len(y_train)]
y_test = y_onehot[len(y_train):] 
#위에 y_train, y_test 의 구간을 다시 나누어야지 onehot인코딩이 됨.

############################### x, y, 합치기 ################################
batch_size = 128
xy_train = train_datagen2.flow(x_train, y_train, 
                               batch_size=128, 
                               shuffle=True)
np.save(save_path + 'keras58_1_cifar10_x_train.npy', arr = x_train)
np.save(save_path + 'keras58_1_cifar10_x_test.npy', arr = x_test)
np.save(save_path + 'keras58_1_cifar10_y_train.npy', arr = y_train)
np.save(save_path + 'keras58_1_cifar10_y_test.npy', arr = y_test)