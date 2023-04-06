import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
save_path = 'd:/study_data/_save/old_young/'
st =time.time()

old_young_x_train = np.load(save_path + 'old_young_x_train.npy')
old_young_x_test = np.load(save_path + 'old_young_x_test.npy')
old_young_y_train = np.load(save_path + 'old_young_y_train.npy')
old_young_y_test = np.load(save_path + 'old_young_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), activation= 'relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(6,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=500)
hist = model.fit(old_young_x_train, old_young_y_train, epochs = 5000, validation_data=(old_young_x_test, old_young_y_test),
                 callbacks = [es])

ed = time.time()
print(ed - st ,2)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(old_young_x_test, old_young_y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(old_young_x_test),axis = 1)
y_test = np.argmax(old_young_y_test, axis=1)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
