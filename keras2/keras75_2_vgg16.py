import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3), )

vgg16.trainable = False # 가중치 동결.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# Trainable : True  30, 30
# Trainable : False 30, 0
# vgg16.trainable = False 30, 4

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688

#  flatten (Flatten)           (None, 512)               0

#  dense (Dense)               (None, 100)               51300

#  dense_1 (Dense)             (None, 10)                1010

# =================================================================
# Total params: 14,766,998
# Trainable params: 14,766,998
# Non-trainable params: 0
# _________________________________________________________________

# 3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', patience=100, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()

# 걸린 시간 계산
elapsed_time = end - start

# 분과 초로 변환
minutes = elapsed_time // 60
seconds = elapsed_time % 60

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')
# 출력
print("걸린 시간: {}분 {}초".format(int(minutes), int(seconds)))