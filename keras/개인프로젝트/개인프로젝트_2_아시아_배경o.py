import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import StratifiedShuffleSplit

path = 'd:/study_data/_data/asian_data1/'
save_path = 'd:/study_data/_save/asian_data1/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
asian_data = datagen.flow_from_directory(path,
            target_size=(128,128),
            batch_size=12000,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

asian_data_x = asian_data[0][0]
asian_data_y = asian_data[0][1]


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

# young_old_x_train, young_old_x_test, young_old_y_train, young_old_y_test = train_test_split(
# young_old_x, young_old_y, train_size= 0.7, shuffle= True, random_state=1557
# )

print(x_train.shape) #(7049, 150, 150, 3)
print(x_test.shape)  #(3021, 150, 150, 3)
print(y_train.shape) #(7049, 79)
print(y_test.shape)  #(3021, 79)

np.save(save_path + 'young_old_x_train.npy', arr = x_train)
np.save(save_path + 'young_old_x_test.npy', arr = x_test)
np.save(save_path + 'young_old_y_train.npy', arr = y_train)
np.save(save_path + 'young_old_y_test.npy', arr = y_test)


