import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
rps = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=900,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

rps_x = rps[0][0]
rps_y = rps[0][1]

end = time.time()
print(end - start, 2)
rps_x_train, rps_x_test, rps_y_train, rps_y_test = train_test_split(
    rps_x, rps_y, train_size= 0.7, shuffle= True, random_state=1557
)

print(rps_x_train.shape) #(630, 150, 150, 3)
print(rps_x_test.shape)  #(270, 150, 150, 3)
print(rps_y_train.shape) #(630, 3)
print(rps_y_test.shape)  #(270, 3)

np.save(save_path + 'keras56_rps_x_train.npy', arr = rps_x_train)
np.save(save_path + 'keras56_rps_x_test.npy', arr = rps_x_test)
np.save(save_path + 'keras56_rps_y_train.npy', arr = rps_y_train)
np.save(save_path + 'keras56_rps_y_test.npy', arr = rps_y_test)

#maxpooling 과적합방지.
#복습