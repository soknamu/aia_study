import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from catboost import CatBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score

# GPU 메모리 증가 코드
physical_devices = tf.config.list_physical_devices('GPU') #gpu상태
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e) 

#1. 데이터
save_path = 'd:/study_data/_save/asian_data/'

x_train = np.load(save_path + 'x_train3.npy', allow_pickle=True)
x_test = np.load(save_path + 'x_test3.npy', allow_pickle=True)
y_train = np.load(save_path + 'y_train3.npy', allow_pickle=True)
y_test = np.load(save_path + 'y_test3.npy', allow_pickle=True)

y_train = y_train.flatten()
y_test = y_test.flatten()
y_train = y_train[:6328]

# 2. 모델
model = CatBoostClassifier(verbose= 0)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
loss = model.score(x_test, y_test)
print('accuracy : ', loss)
