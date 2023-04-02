import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/horse-or-human/'
save_path = 'd:/study_data/_save/horse-or-human/'
datagen = ImageDataGenerator(rescale = 1./255)

start = time.time()
human_horse = datagen.flow_from_directory('d:/study_data/_data/horse-or-human/',
        target_size=(150, 150),
        batch_size= 15000,
        class_mode= 'binary',
        color_mode = 'rgb',
        shuffle= True)

human_horse_x = human_horse[0][0]
human_horse_y = human_horse[0][1]


end = time.time()
print(end - start, 2)
human_horse_x_train, human_horse_x_test, human_horse_y_train, human_horse_y_test = train_test_split(
    human_horse_x, human_horse_y, train_size=0.7, shuffle= True, random_state= 1503
)

print(human_horse_x_train.shape) #(718, 150, 150, 3)
print(human_horse_x_test.shape)  #(309, 150, 150, 3)
print(human_horse_y_train.shape) #(718,)
print(human_horse_y_test.shape)  #(309,)

np.save(save_path + 'keras56_human_horse_x_train.npy', arr = human_horse_x_train)
np.save(save_path + 'keras56_human_horse_x_test.npy', arr = human_horse_x_test)
np.save(save_path + 'keras56_human_horse_y_train.npy', arr = human_horse_y_train)
np.save(save_path + 'keras56_human_horse_y_test.npy', arr = human_horse_y_test)

