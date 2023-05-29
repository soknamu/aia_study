import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터
path = 'c:/study/_data/cat_dog/PetImages/'
save_path = 'c:/study/_save/cat_dog/'
train_datagen = ImageDataGenerator(
    rescale= 1./255, #스케일링 하겠다.(. 붙히는 이유 : 부동소수점으로 연산, 정규화(nonaliazation))
    # horizontal_flip=True, #(가로뒤집기)
    # vertical_flip= True,  #좌우 반전
    # width_shift_range= 0.1, #소수점만큼의 사진을 넓이를 이동시킨다.(증폭)
    # height_shift_range= 0.1, # 소수점만큼의 사진을 높이를 이동시킨다.(증폭)
    # rotation_range= 5,
    # zoom_range= 1.2, #확대
    # shear_range= 0.7, #찌그러 트리는거
    # fill_mode= 'nearest',  # 이동된자리가 짤림 그걸방지하는 것이 있음. nearest는 옮겨진 값을 근처값으로 넣어줌.
    # -> 지금까지 기능은 다 증폭기능. 넣거나 더 빼도됨.
)

# test_datagen = ImageDataGenerator(
#     rescale= 1./255, )  #평가 데이터를 증폭시키는 것은 데이터조작이 될 수 있어서 증폭시키지 않음.


xy_train = train_datagen.flow_from_directory(
    'c:/study/_data/cat_dog/PetImages/', #ad normal 폴더로 들어가지 않는 이유. 라벨값이 정해져 있기 때문에. 그래서 상위폴더로 지정해줌.
    target_size=(169, 169), #모은 사진들을 확대 또는 축소를 해서 크기를 무조건 200 * 200으로 만듬. (크든 작든)
    batch_size= 24998,  #5장씩 쓰겠다. #전체데이터를 쓸라면 160(데이터의 최대의 개수만큼(그이상도 가능.))넣어라!
    class_mode= 'binary',#데이터가 2개밖에 없기때문에 (수치화 되서 만들어줌) 카테고리컬은
    color_mode= 'rgb',#흑백칼라 
    #color_mode= 'rgba',#빨초파(투명도)
    shuffle= True,) 
#directory = folder

# xy_test = test_datagen.flow_from_directory(
#     'd:/study_data/_data/cat_dog/PetImages/',
#     target_size=(300, 300),
#     batch_size= 225000,
#     class_mode= 'binary',
#     color_mode= 'rgb',
#     shuffle= True,)


import time
start_time = time.time()
np.save(save_path + 'keras56_x_train.npy', arr = xy_train[0][0])
# np.save(save_path + 'keras56_1_x_test.npy', arr = xy_test[0][0])
np.save(save_path + 'keras56_y_train.npy', arr = xy_train[0][1])
# np.save(save_path + 'keras56_1_y_test.npy', arr = xy_test[0][1])
end_time = time.time()
print("걸린시간 : ", round(end_time - start_time, 2))
#걸린시간 :  774.43
