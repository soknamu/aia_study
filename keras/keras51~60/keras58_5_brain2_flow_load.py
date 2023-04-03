import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import time
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
save_path = 'd:/study_data/_save/brain/'
st =time.time()

x_train = np.load(save_path + 'keras58_brain_x_train.npy')
x_test = np.load(save_path + 'keras58_brain_x_test.npy')
y_train = np.load(save_path + 'keras58_brain_y_train.npy')
y_test = np.load(save_path + 'keras58_brain_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,1), activation= 'relu')) #x_train.shape[1:]
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#y_train.shape[1] = output 의 값.
#3.컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=50)
hist = model.fit(x_train, y_train, epochs = 50, validation_data=(x_test, y_test),
                 callbacks = [es])

ed = time.time()
print(ed - st ,2)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = np.round(model.predict(x_test))
y_test = np.round(y_test)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)