import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import time

#1. 데이터
save_path = 'd:/study_data/_save/men_women/'
st =time.time()

men_women_x_train = np.load(save_path + 'keras58_men_women_x_train.npy')
men_women_x_test = np.load(save_path + 'keras58_men_women_x_test.npy')
men_women_y_train = np.load(save_path + 'keras58_men_women_y_train.npy')
men_women_y_test = np.load(save_path + 'keras58_men_women_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(32, (2,3), input_shape=(150,150,3), activation= 'relu'))
model.add(Conv2D(32, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
hist = model.fit(men_women_x_train, men_women_y_train, epochs = 34, validation_data=(men_women_x_test, men_women_y_test))

ed = time.time()
print(ed - st, 2)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(men_women_x_test, men_women_y_test)
print('loss : ', loss)

y_pred = np.round(model.predict(men_women_x_test))

from sklearn.metrics import accuracy_score

acc = accuracy_score(men_women_y_test, y_pred)
print('acc : ', acc)


# loss :  [3.8113880157470703, 0.5066666603088379]
# acc :  0.5066666666666667

#복습