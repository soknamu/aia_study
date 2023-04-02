#수치로 되있는 데이터를 증폭시킨다 =flow

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
    vertical_flip=True,
    width_shift_range= 0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode= 'nearest'
)

augment_size = 100 #증폭사이즈.
print(x_train.shape)       #(60000, 28, 28) 6만장 , 28*28
print(x_train[0].shape)    #(28, 28) -> 0번째 사진의 모양이니깐 28*28
print(x_train[1].shape)    #(28, 28) 0과 동일함.
print(x_train[0][0].shape) #(28,)

print(np.tile(x_train[0].reshape(28*28),  #np.tile 의 수치를 늘리기 위해서 리쉐이프 모양으로 함.
              augment_size).reshape(-1,28,28,1).shape) #tile 비슷한 걸로 계속 복사시킴.
#(100, 28, 28, 1)
#트레인의 0번쨰를 augment_size 만큼 증폭해서 (-1,28,28,1) 모양으로 리쉐이프 해라.
#np.tile(데이터, 증폭시킬갯수)

print(np.zeros(augment_size)) #-> 100개의 0을 출력해준다.
print(np.zeros(augment_size).shape) #(100,)

x_data = train_datagen.flow(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1), # x의 데이터
                            np.zeros(augment_size), #y의 데이터 : 그림만 그릴꺼라 필요없어서 걍  0넣어줌.
                            batch_size=augment_size,
                            shuffle=True,
) 
#flow와 디렉토리의 차이.
#차이점 : 경로를 받아드리지 않는다.
print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x00000262D3EB4FA0> 이터레이터 뜸.

print(x_data[0]) #x와 y가 모두 포함.
print(x_data[0][0].shape) #(100, 28, 28, 1) 모든 데이터의 x
print(x_data[0][1].shape) #(100,) 모든 데이터의 y

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1) #총 49개의 플롯을 만들고,
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap= 'gray')
plt.show()
