import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
path = 'c:/study_data/_data/men_women/'
save_path = 'c:study_data/_save/men_women/'
datagen = ImageDataGenerator(rescale= 1./255)

start = time.time()
men_women = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=2000,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

men_women_x = men_women[0][0]
men_women_y = men_women[0][1]

end = time.time()
print(end - start, 2)
men_women_x_train, men_women_x_test, men_women_y_train, men_women_y_test = train_test_split(
    men_women_x, men_women_y, train_size= 0.7, shuffle= True, random_state=1557
)

print(men_women_x_train.shape) #(1400, 150, 150, 3)
print(men_women_x_test.shape)  #(600, 150, 150, 3)
print(men_women_y_train.shape) #(1400, 2)
print(men_women_y_test.shape)  #(600, 2)

np.save(save_path + 'keras56_men_women_x_train.npy', arr = men_women_x_train)
np.save(save_path + 'keras56_men_women_x_test.npy', arr = men_women_x_test)
np.save(save_path + 'keras56_men_women_y_train.npy', arr = men_women_y_train)
np.save(save_path + 'keras56_men_women_y_test.npy', arr = men_women_y_test)
