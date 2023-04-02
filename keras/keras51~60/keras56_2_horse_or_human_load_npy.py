import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#1. 데이터
save_path = 'd:/study_data/_save/horse-or-human/'

horse_human_x_train = np.load(save_path + 'keras56_human_horse_x_train.npy')
horse_human_x_test = np.load(save_path + 'keras56_human_horse_x_test.npy')
horse_human_y_train = np.load(save_path + 'keras56_human_horse_y_train.npy')
horse_human_y_test = np.load(save_path + 'keras56_human_horse_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(32, (2,3), input_shape=(150,150,3), activation= 'relu'))
model.add(Conv2D(32, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
hist = model.fit(horse_human_x_train, horse_human_y_train, epochs = 34, validation_data=(horse_human_x_test, horse_human_y_test))

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(horse_human_x_test, horse_human_y_test)
print('loss : ', loss)

y_pred = np.round(model.predict(horse_human_x_test))

from sklearn.metrics import accuracy_score

acc = accuracy_score(horse_human_y_test, y_pred)
print('acc : ', acc)

# loss :  [0.024048959836363792, 0.9902912378311157]
# acc :  0.9902912621359223