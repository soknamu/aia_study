
#수치로 되있는 데이터를 증폭시킨다 =flow

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
#0. seed initialization

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)
# -> 이것을 통해서 랜덤하게 들어가는 값을 고정시킴

#또는


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

#randidx = np.random.randint(60000, size = 40000) #
randidx = np.random.randint(x_train.shape[0], size = augment_size)
#위에 식이랑 아래 식이랑 같음.

print(randidx) #[48154 52252 17050 ... 21077 53138 19075] 이게 돌릴때마다 계속바뀌어서 정확도를 찾기 어려움 그래서 random 시드를 사용/
print(randidx.shape) #(40000,)
print(np.min(randidx), np.max(randidx)) #0 59995 랜덤으로 들어간 값.

x_augmented = x_train[randidx].copy() # x_augmented 에 4만개가 들어감. copy를 통해서 x_train데이터가 덮어씌어지지 않음.
y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape, y_augmented.shape) #(40000,28,28) , (40000,)

#x_augmented랑 x_train이랑 합쳐야됨.

#차원 바꾸기(3-> 4)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                       x_test.shape[1], 
                       x_test.shape[2], 1)
#x_test = x_test.shape(10000, 28, 28, 1) 와 동일.

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

#list [1,1] +[1] -> [1,1,1] 차원변경가능. numpy는 안됨.-> [1,1] +[1,1] => [2,2]

#변환
# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented, #y는 넣을 필요는 없는데 쌍으로 되있어서 넣음 (잘모름)
# batch_size=augment_size, shuffle= False
# ) -> next 안쓴것.

#print(x_augmented) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000002033554A0D0>
# print(x_augmented[0][0].shape) #(40000, 28, 28, 1)


x_augmented = train_datagen.flow(
    x_augmented, y_augmented, #y는 넣을 필요는 없는데 쌍으로 되있어서 넣음 (잘모름)
batch_size=augment_size, shuffle= False
).next()[0] #했을 때 튜플이 나옴 즉, x_augmented[0]가 나옴 그래서 [0]을 붙혀줌. 그러면 x_augmented[0][0]가 나옴.

# print(x_augmented)
# print(x_augmented.shape) #(40000, 28, 28)

print(np.max(x_train), np.min(x_train)) #255 0
print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

#x_train = x_train + x_augmented
x_train = np.concatenate((x_train/255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
x_test = x_test/255.

print(x_train.shape)
print(y_train.shape)

print(np.max(x_train), np.min(x_train)) #1.0 0.0

#복습.