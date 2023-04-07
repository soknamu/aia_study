import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization
import time
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
save_path = 'd:/study_data/_save/asian_data/'
st =time.time()

asian_data_x_train = np.load(save_path + 'asian_data_x_train.npy')
asian_data_x_test = np.load(save_path + 'asian_data_x_test.npy')
asian_data_y_train = np.load(save_path + 'asian_data_y_train.npy')
asian_data_y_test = np.load(save_path + 'asian_data_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(200,200,3), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation= 'relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(79,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=500)
hist = model.fit(asian_data_x_train, asian_data_y_train, epochs = 50, validation_data=(asian_data_x_test, asian_data_y_test),
                 callbacks = [es])

ed = time.time()
print(np.round(ed - st ,2))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(asian_data_x_test, asian_data_y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(asian_data_x_test),axis = 1)
y_test = np.argmax(asian_data_y_test, axis=1)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
