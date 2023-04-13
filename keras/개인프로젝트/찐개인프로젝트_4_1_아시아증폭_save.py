import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import StratifiedShuffleSplit

path = 'd:/study_data/_data/asian_data/'
save_path = 'd:/study_data/_save/asian_data/'
datagen = ImageDataGenerator(rescale=1.,
                             horizontal_flip=True,
                             zoom_range= 1.2,)

start = time.time()
asian_data = datagen.flow_from_directory(path,
            target_size=(96,96),
            batch_size=9040,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

asian_data_x = asian_data[0][0]
asian_data_y = asian_data[0][1]
augment_size = 2960
print(f'runtime : {time.time()-start}')

# StratifiedShuffleSplit을 사용하여 데이터를 라벨별로 분리합니다.
split = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
for train_index, test_index in split.split(asian_data_x, asian_data_y):
    train_index = train_index.flatten()
    test_index = test_index.flatten()

    x_train = asian_data_x[train_index]
    y_train = asian_data_y[train_index]
    x_test = asian_data_x[test_index]
    y_test = asian_data_y[test_index]

randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False).next()[0]

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)
x_test = x_test/255.

print(x_train.shape) #(9288, 96, 96, 3)
print(x_test.shape)  #(2712, 96, 96, 3)
print(y_train.shape) #(9288, 6)
print(y_test.shape)  #(2712, 6)

np.save(save_path + 'x_train1.npy', arr = x_train)
np.save(save_path + 'x_test1.npy', arr = x_test)
np.save(save_path + 'y_train1.npy', arr = y_train)
np.save(save_path + 'y_test1.npy', arr = y_test)


