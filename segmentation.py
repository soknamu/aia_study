import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

# MNIST 데이터셋 로드
(x_train, _), (x_test, _) = mnist.load_data()

# 데이터 전처리
x_train = np.expand_dims(x_train, axis=-1)  # 차원 확장
x_test = np.expand_dims(x_test, axis=-1)
x_train = x_train.astype('float32') / 255.0  # 0 ~ 1 사이로 정규화
x_test = x_test.astype('float32') / 255.0

# 입력 이미지 크기 조정
x_train_resized = tf.image.resize(x_train, [112, 112])
x_test_resized = tf.image.resize(x_test, [112, 112])

# 타깃 이미지 크기 조정
x_train_target = tf.image.resize(x_train, [56, 56])
x_test_target = tf.image.resize(x_test, [56, 56])

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(56, 56, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy')

# 모델 학습
model.fit(x_train_resized, x_train_target, epochs=10, batch_size=32, validation_data=(x_test_resized, x_test_target))

# 예측
preds = model.predict(x_test_resized)

# 예측 결과 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 2, figsize=(10, 20))
for i, ax in enumerate(axes):
    # 입력 이미지
    ax[0].imshow(x_test_resized[i].squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    
    # 세분화 결과
    ax[1].imshow(preds[i].squeeze(), cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Segmentation Output')

plt.tight_layout()
plt.show()
