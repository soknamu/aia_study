import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'
datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip=True,
    vertical_flip= True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range= 1.2,
    shear_range= 0.7,
    fill_mode= 'nearest',  
   
)

test_datagen2 = ImageDataGenerator(rescale= 1./1)
augment_size = 2500


rps = datagen.flow_from_directory(path,
            target_size=(150,150),
            batch_size=900,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

rps_x = rps[0][0]
rps_y = rps[0][1]

rps_x_train, rps_x_test, rps_y_train, rps_y_test = train_test_split(
    rps_x, rps_y, train_size= 0.7, shuffle= True, random_state=1557
)

randidx = np.random.randint(rps_x_train.shape[0], size = augment_size)

x_augmented = rps_x_train[randidx].copy() # x_augmented 에 4만개가 들어감. copy를 통해서 x_train데이터가 덮어씌어지지 않음.
y_augmented = rps_y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False).next()[0]

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train = np.concatenate((rps_x_train/255., x_augmented), axis=0)
y_train = np.concatenate((rps_y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
x_test = rps_x_test/255.

print(rps_x_train.shape) #(630, 150, 150, 3)
print(rps_x_test.shape)  #(270, 150, 150, 3)
print(rps_y_train.shape) #(630, 3)
print(rps_y_test.shape)  #(270, 3)

np.save(save_path + 'keras58_rps_x_train.npy', arr = rps_x_train)
np.save(save_path + 'keras58_rps_x_test.npy', arr = rps_x_test)
np.save(save_path + 'keras58_rps_y_train.npy', arr = rps_y_train)
np.save(save_path + 'keras58_rps_y_test.npy', arr = rps_y_test)

#maxpooling 과적합방지.