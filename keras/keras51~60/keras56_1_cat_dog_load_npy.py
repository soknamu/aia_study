import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 전처리 데이터
from sklearn.model_selection import train_test_split
import time
#1. 데이터

path = 'd:/study_data/_save/cat_dog/'
# np.save(path + 'keras55_1_x_train.npy', arr = xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr = xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr = xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr = xy_test[0][1])

st = time.time()
x = np.load(path + 'keras56_x_train.npy')
y = np.load(path + 'keras56_y_train.npy')
et = time.time()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, 
                                                    random_state= 123, shuffle= True)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape= (250,250,3), activation= 'relu'))
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs=1000, 
                    steps_per_epoch= 32,
                    validation_split = 0.2,
                    validation_steps=24,
                    )

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print(acc)

print('val_loss : ', val_loss[-1])
print('loss : ', loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = np.round(model.predict(x_test))

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

#1. 그림그리기!
from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(loss,label= 'loss')
plt.plot(val_loss,label= 'val_loss')

plt.subplot(1,2,2)
plt.plot(acc,label= 'acc')
plt.plot(val_acc,label= 'val_acc')

plt.show()