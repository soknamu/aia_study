import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

tf.random.set_seed(337)
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32,32,3), )

vgg16.trainable = True # 가중치 동결. #false가 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# 3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(monitor='val_acc', mode='max', patience=100, verbose=1, restore_best_weights=True)
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
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
# acc : 0.5842
# 걸린 시간: 8분 29초

#flatten 가중치 동결(True)
# acc 0.8047000169754028
# acc : 0.8047
# 걸린 시간: 17분 52초

#average 가중치 동결(False)
# acc : 0.5842
# 걸린 시간: 8분 3초

#average 가중치 동결X(True)
# acc 0.8181999921798706
# acc : 0.8182
# 걸린 시간: 18분 23초

