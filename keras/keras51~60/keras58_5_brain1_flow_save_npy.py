import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터
from sklearn.model_selection import train_test_split
path = 'd:/study_data/_data/brain/'
save_path = 'd:/study_data/_save/brain/'

datagen = ImageDataGenerator(
    rescale= 1./255, #스케일링 하겠다.(. 붙히는 이유 : 부동소수점으로 연산, 정규화(nonaliazation))
    horizontal_flip=True, #(가로뒤집기)
    vertical_flip= True,  #좌우 반전
    width_shift_range= 0.1, #소수점만큼의 사진을 넓이를 이동시킨다.(증폭)
    height_shift_range= 0.1, # 소수점만큼의 사진을 높이를 이동시킨다.(증폭)
    rotation_range= 5,
    zoom_range= 1.2, #확대
    shear_range= 0.7, #찌그러 트리는거
    fill_mode= 'nearest',  # 이동된자리가 짤림 그걸방지하는 것이 있음. nearest는 옮겨진 값을 근처값으로 넣어줌.
    # -> 지금까지 기능은 다 증폭기능. 넣거나 더 빼도됨.
)

test_datagen2 = ImageDataGenerator(
    rescale= 1./1,   #평가 데이터를 증폭시키는 것은 데이터조작이 될 수 있어서 증폭시키지 않음.
)

augment_size = 25000


brain = datagen.flow_from_directory(path,
        target_size=(150, 150),
        batch_size= 15000,
        class_mode= 'binary',
        color_mode = 'grayscale',
        shuffle= True)

brain_x = brain[0][0]
brain_y = brain[0][1]

brain_x_train, brain_x_test, brain_y_train, brain_y_test = train_test_split(
    brain_x, brain_y, train_size=0.7, shuffle= True, random_state= 1503
)

randidx = np.random.randint(brain_x_train.shape[0], size = augment_size)

x_augmented = brain_x_train[randidx].copy() # x_augmented 에 4만개가 들어감. copy를 통해서 x_train데이터가 덮어씌어지지 않음.
y_augmented = brain_y_train[randidx].copy()

x_augmented = datagen.flow(
    x_augmented, y_augmented,
batch_size=augment_size, shuffle= False
).next()[0]

# print(np.max(x_train), np.min(x_train)) #255 0
# print(np.max(x_augmented), np.min(x_augmented)) #1.0 0.0

x_train = np.concatenate((brain_x_train/255., x_augmented), axis=0)
y_train = np.concatenate((brain_y_train, y_augmented), axis=0) #y스케일링하면 큰일남.
x_test = brain_x_test/255.

print(brain_x_train.shape) #(718, 150, 150, 3)
print(brain_x_test.shape)  #(309, 150, 150, 3)
print(brain_y_train.shape) #(718,)
print(brain_y_test.shape)  #(309,)

np.save(save_path + 'keras58_brain_x_train.npy', arr = brain_x_train)
np.save(save_path + 'keras58_brain_x_test.npy', arr = brain_x_test)
np.save(save_path + 'keras58_brain_y_train.npy', arr = brain_y_train)
np.save(save_path + 'keras58_brain_y_test.npy', arr = brain_y_test)
