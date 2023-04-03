import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import time

#1. 데이터
save_path = 'd:/study_data/_save/dog_breed/'

dog_breed_x_train = np.load(save_path + 'keras58_dog_breed_x_train.npy')
dog_breed_x_test = np.load(save_path + 'keras58_dog_breed_x_test.npy')
dog_breed_y_train = np.load(save_path + 'keras58_dog_breed_y_train.npy')
dog_breed_y_test = np.load(save_path + 'keras58_dog_breed_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(32, (2,3), input_shape=(150,150,4), activation= 'relu'))
model.add(Conv2D(32, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(5,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
hist = model.fit(dog_breed_x_train, dog_breed_y_train, epochs = 34, validation_data=(dog_breed_x_test, dog_breed_y_test))


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(dog_breed_x_test, dog_breed_y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(dog_breed_x_test),axis = 1)
y_test = np.argmax((dog_breed_y_test),axis = 1)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

