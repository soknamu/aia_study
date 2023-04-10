import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/asian_data/'
save_path = 'd:/study_data/_save/asian_data/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
young_old = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=10070,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

young_old_x = young_old[0][0]
young_old_y = young_old[0][1]

end = time.time()
print(end - start, 2)
young_old_x_train, young_old_x_test, young_old_y_train, young_old_y_test = train_test_split(
young_old_x, young_old_y, train_size= 0.7, shuffle= True, random_state=1557
)

print(young_old_x_train.shape) #(7049, 150, 150, 3)
print(young_old_x_test.shape)  #(3021, 150, 150, 3)
print(young_old_y_train.shape) #(7049, 79)
print(young_old_y_test.shape)  #(3021, 79)

np.save(save_path + 'young_old_x_train.npy', arr = young_old_x_train)
np.save(save_path + 'young_old_x_test.npy', arr = young_old_x_test)
np.save(save_path + 'young_old_y_train.npy', arr = young_old_y_train)
np.save(save_path + 'young_old_y_test.npy', arr = young_old_y_test)