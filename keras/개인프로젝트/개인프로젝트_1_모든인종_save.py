import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'd:/study_data/_data/old_young/'
save_path = 'd:/study_data/_save/old_young/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
old_young = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=900,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

old_young_x = old_young[0][0]
old_young_y = old_young[0][1]

end = time.time()
print(end - start, 2)
old_young_x_train, old_young_x_test, old_young_y_train, old_young_y_test = train_test_split(
old_young_x, old_young_y, train_size= 0.7, shuffle= True, random_state=1557,stratify= old_young_y
)

print(old_young_x_train.shape) #(598, 150, 150, 3)
print(old_young_x_test.shape)  #(257, 150, 150, 3)
print(old_young_y_train.shape) #(598, 6)
print(old_young_y_test.shape)  #(257, 6)

np.save(save_path + 'old_young_x_train.npy', arr = old_young_x_train)
np.save(save_path + 'old_young_x_test.npy', arr = old_young_x_test)
np.save(save_path + 'old_young_y_train.npy', arr = old_young_y_train)
np.save(save_path + 'old_young_y_test.npy', arr = old_young_y_test)