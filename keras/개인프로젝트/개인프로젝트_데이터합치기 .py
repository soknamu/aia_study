import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

path = 'd:/study_data/_data/asian_data/'
datagen = ImageDataGenerator(rescale=1.)

asian_data = datagen.flow_from_directory(path,
            target_size=(96,96),
            batch_size=9040,
            class_mode='categorical',
            color_mode= 'rgb',
            shuffle= True)

asian_data_x = asian_data[0][0]
asian_data_y = asian_data[0][1]

# StratifiedShuffleSplit을 사용하여 데이터를 라벨별로 분리합니다.
split = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
for train_index, test_index in split.split(asian_data_x, asian_data_y):
    train_index = train_index.flatten()
    test_index = test_index.flatten()

    x_train = asian_data_x[train_index]
    y_train = asian_data_y[train_index]
    x_test = asian_data_x[test_index]
    y_test = asian_data_y[test_index]

x_train = x_train/255.
x_test = x_test/255.

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

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, restore_best_weights=True, patience=1)
hist = model.fit(x_train, y_train, epochs = 50, batch_size= 25, validation_data=(x_test, y_test),
                 callbacks = [es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = np.argmax(model.predict(x_test),axis = 1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

pred_path = 'd:/study_data/_data/asian_data_predict/' 
predict = datagen.flow_from_directory(pred_path, target_size = (96,96), class_mode = 'categorical',color_mode='RGB', shuffle=False)

x_pred = predict[0][0]
predict_age = predict[0][1]

y_pred = np.argmax(model.predict(x_pred),axis = 1)

print('나이는 :', np.argmax(predict_age,axis=1))

print('acc : ', acc(np.argmax(predict_age, axis=1),y_pred))

