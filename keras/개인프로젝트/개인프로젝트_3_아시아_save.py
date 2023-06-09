import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import StratifiedShuffleSplit

path = 'd:/study_data/_data/asian_data/'
save_path = 'd:/study_data/_save/asian_data/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
asian_data = datagen.flow_from_directory(path,
            target_size=(300,300),
            batch_size=10070,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

asian_data_x = asian_data[0][0]
asian_data_y = asian_data[0][1]

print(f'runtime : {time.time()-start}')

# StratifiedShuffleSplit을 사용하여 데이터를 라벨별로 분리합니다.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=42)
for train_index, test_index in split.split(asian_data_x, asian_data_y):
    x_train = [asian_data_x[i] for i in train_index]
    y_train = [asian_data_y[i] for i in train_index]
    x_test = [asian_data_x[i] for i in test_index]
    y_test = [asian_data_y[i] for i in test_index]

print(x_train.shape) #(6479, 128, 128, 3)
print(x_test.shape)  #(2778, 128, 128, 3)
print(y_train.shape) #(6479, 59)
print(y_test.shape)  #(2778, 59)

np.save(save_path + 'x_train.npy', arr = x_train)
np.save(save_path + 'x_test.npy', arr = x_test)
np.save(save_path + 'y_train.npy', arr = y_train)
np.save(save_path + 'y_test.npy', arr = y_test)