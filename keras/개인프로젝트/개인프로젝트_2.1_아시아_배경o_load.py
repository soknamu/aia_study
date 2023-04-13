import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Input
import time
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') #gpu상태
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)    


#1. 데이터
save_path = 'd:/study_data/_save/asian_data1/'
st =time.time()

x_train = np.load(save_path + 'x_train.npy')
x_test = np.load(save_path + 'x_test.npy')
y_train = np.load(save_path + 'y_train.npy')
y_test = np.load(save_path + 'y_test.npy')

#2.모델구성
model = Sequential()
model.add(Input(shape=(128,128,3)))
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
model.add(Conv2D(512, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dense(59,activation='softmax'))
model.summary()

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=50)
hist = model.fit(x_train, y_train, epochs = 5000, batch_size= 16, validation_data=(x_test, y_test),
                 callbacks = [es])

print(f'runtime : {time.time()-st}')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(x_test),axis = 1)
y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
