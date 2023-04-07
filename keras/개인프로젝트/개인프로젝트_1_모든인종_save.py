import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/asian_data/'
save_path = 'd:/study_data/_save/asian_data/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
asian_data = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=12200,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

asian_data_x = asian_data[0][0]
asian_data_y = asian_data[0][1]

end = time.time()
print(round(end - start, 2))
asian_data_x_train, asian_data_x_test, asian_data_y_train, asian_data_y_test = train_test_split(
asian_data_x, asian_data_y, train_size= 0.7, shuffle= True, random_state=1557)

print(asian_data_x_train.shape) #(8815, 200, 200, 3)
print(asian_data_x_test.shape)  #(3778, 200, 200, 3)
print(asian_data_y_train.shape) #(8815, 79)
print(asian_data_y_test.shape)  #(3778, 79)

np.save(save_path + 'asian_data_x_train.npy', arr = asian_data_x_train)
np.save(save_path + 'asian_data_x_test.npy', arr = asian_data_x_test)
np.save(save_path + 'asian_data_y_train.npy', arr = asian_data_y_train)
np.save(save_path + 'asian_data_y_test.npy', arr = asian_data_y_test)