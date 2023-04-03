import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import time

#1. 데이터
save_path = 'd:/study_data/_save/cat_dog/'
st =time.time()

cat_dog_x_train = np.load(save_path + 'keras58_cat_dog_x_train.npy')
cat_dog_x_test = np.load(save_path + 'keras58_cat_dog_x_test.npy')
cat_dog_y_train = np.load(save_path + 'keras58_cat_dog_y_train.npy')
cat_dog_y_test = np.load(save_path + 'keras58_cat_dog_y_test.npy')

#2.모델구성
model = Sequential()
model.add(Conv2D(32, (2,3), input_shape=(150,150,3), activation= 'relu'))
model.add(Conv2D(32, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
hist = model.fit(cat_dog_x_train, cat_dog_y_train, epochs = 34, validation_data=(cat_dog_x_test, cat_dog_y_test))

ed = time.time()
print(ed - st, 2)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(cat_dog_x_test, cat_dog_y_test)
print('loss : ', loss)

y_pred = np.round(model.predict(cat_dog_x_test))
y_test = np.round((cat_dog_y_test))
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)


# loss :  [0.0842825323343277, 0.9814814925193787]
# acc :  0.9814814814814815