import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

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

#2.모델구성
model = Sequential()
model.add(Input(shape=(96,96,3)))
model.add(Conv2D(64, (2,2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.summary()

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=120)
hist = model.fit(x_train, y_train, epochs = 50000, batch_size= 30
                 , validation_data=(x_test, y_test),
                 callbacks = [es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

datagen = ImageDataGenerator()
pred_path = 'd:/study_data/_data/asian_data_predict/' 
predict = datagen.flow_from_directory(pred_path, target_size=(96,96), class_mode='categorical', color_mode='rgb', shuffle=False)
0.
x_pred = predict[0][0]
predict_age = np.argmax(predict[0][1],axis =1)
y_pred = np.argmax(model.predict(x_pred),axis = 1)

print('x_pred.shape:', x_pred.shape) #(12, 96, 96, 3)
print('predict_age.shape:', predict_age.shape) #(12,)

print('실제 나이는 :',predict_age,'\n예측한 나이 : ',y_pred)
print('acc: ', accuracy_score(predict_age,y_pred))



'''
#2.모델구성
model = Sequential()
model.add(Input(shape=(96,96,3)))
model.add(Conv2D(128, (2,2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, (2,2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(512, (3,3), padding='valid', activation='relu'))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.summary()
'''
