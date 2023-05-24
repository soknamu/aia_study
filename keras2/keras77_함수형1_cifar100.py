 # 함수형 맹그러봐!
import time
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

tf.random.set_seed(337)
(x_train,y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(weights='imagenet', include_top=False)(input1)
gap1 = GlobalAveragePooling2D()(vgg16)
# flt = Flatten()(vgg16)
# hidden1 = Dense(100)(flt)
hidden1 = Dense(100)(gap1)
output1 = Dense(100, activation='softmax')(hidden1)

model = Model(inputs = input1, outputs = output1)

model.summary()

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

# loss : 1.391618251800537
# acc 0.8122000098228455
# acc : 0.8122
# 걸린 시간: 17분 53초