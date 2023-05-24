import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

tf.random.set_seed(337)
(x_train,y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3), )

vgg16.trainable = True # 가중치 동결.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# 3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', patience=100, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
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

#flatten 가중치 동결(False)
# loss : 2.958848714828491
# acc 0.3391000032424927
# acc : 0.3391
# 걸린 시간: 4분 4초

#flatten 가중치 동결(True)

# loss : 4.6056318283081055
# acc 0.009999999776482582
# acc : 0.01
# 걸린 시간: 9분 2초

#average 가중치 동결(False)
# loss : 2.9588541984558105
# acc 0.3391000032424927
# acc : 0.3391
# 걸린 시간: 4분 5초

#average 가중치 동결X(True)
# loss : 4.183261394500732
# acc 0.4025000035762787
# acc : 0.4025
# 걸린 시간: 9분 11초