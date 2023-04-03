#수치로 되있는 데이터를 증폭시킨다 =flow

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
#0. seed initialization

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)
# -> 이것을 통해서 랜덤하게 들어가는 값을 고정시킴

(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
    vertical_flip=True, #뒤집기
    width_shift_range= 0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode= 'nearest'
)

augment_size = 40000 # 4만개를 증폭할꺼다.

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[:augment_size].copy() # 0부터 x_train 짤라서 copy 해줌.   augment까지 짤라준다.
y_augmented = y_train[randidx].copy()

# print(x_augmented)
# print(x_augmented.shape, y_augmented.shape) #(40000,28,28) , (40000,)


#차원 바꾸기(3-> 4)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                       x_test.shape[1], 
                       x_test.shape[2], 1)
#x_test = x_test.shape(10000, 28, 28, 1) 와 동일.

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

#변환

x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False
).next()[0]

print(np.max(x_train), np.min(x_train)) #255 0
print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

# x_train = np.concatenate((x_train/255., x_augmented), axis=0)
# y_train = np.concatenate((y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
# x_test = x_test/255.

print(x_train.shape)
print(y_train.shape)

print(np.max(x_train), np.min(x_train)) #1.0 0.0

# y_train = np.array(pd.get_dummies(y_train))
# y_test = np.array(pd.get_dummies(y_test))

'''
#2.모델구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(28,28,1), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=500)
hist = model.fit(x_train, y_train, epochs = 15, validation_data=(x_test, y_test),batch_size= 256,
                 callbacks = [es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(x_test),axis = 1)
y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
'''
#실습

#x_augmented 10개와 x_train 10개를 비교하는 이미지를 출력할 것!!

print(x_train.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i]/255., cmap= 'gray')
    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')

plt.show()
