import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRFClassifier
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
physical_devices = tf.config.list_physical_devices('GPU') #gpu상태
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e) 

#1. 데이터
save_path = 'd:/study_data/_save/asian_data/'

x_train = np.load(save_path + 'x_train.npy')
x_test = np.load(save_path + 'x_test.npy')
y_train = np.load(save_path + 'y_train.npy')
y_test = np.load(save_path + 'y_test.npy')

# Reshape the training and testing data to be 2D
x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

#2. 모델
model = XGBRFClassifier()

#3.컴파일, 훈련

model.fit(x_train_reshaped, y_train)

#4.평가, 예측
loss = model.score(x_test_reshaped, y_test)
print('loss : ', loss)

datagen = ImageDataGenerator()
pred_path = 'd:/study_data/_data/asian_data_predict/' 
predict = datagen.flow_from_directory(pred_path, target_size=(96,96), class_mode='categorical', color_mode='rgb', shuffle=False)
predict_label = predict.classes
predict_num = predict.samples

x_pred = predict.next()[0].reshape(predict_num, -1)
y_pred = model.predict(x_pred)

print('x_pred.shape:', x_pred.shape) #(12, 27648)
print('predict_label.shape:', predict_label.shape) #(12,)

print('실제 나이는 :',predict_label,'\n예측한 나이 : ',y_pred)
print('acc: ', accuracy_score(predict_label, y_pred))

