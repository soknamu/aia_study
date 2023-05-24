 # 함수형 맹그러봐!
import time
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16,InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

tf.random.set_seed(337)
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape= (32, 32, 3))
# print(base_model.output) #마지막 레이어.
#KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), 
#                                 dtype=tf.float32, name=None), 
#            name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")
x = base_model.output 
x = GlobalAveragePooling2D()(x)
x = Dense(100, activation='relu')(x)
output1 = Dense(10, activation='softmax')(x)

model = Model(inputs = base_model.input, outputs = output1)

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
# loss : 1.2130976915359497
# acc 0.8151000142097473
# acc : 0.8151
# 걸린 시간: 18분 46초