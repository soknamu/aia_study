import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
# 1. 데이터
x, y = fetch_covtype(return_X_y=True)
# One-hot 인코딩
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1,1))

print(x.shape, y.shape) #(581012, 54) (581012, 7)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, train_size=0.8, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.random.normal([54, 50], dtype=tf.float32), name='weight')
b1 = tf.compat.v1.Variable(tf.zeros([50], dtype=tf.float32), name='bias')
layer1 = tf.compat.v1.matmul(x, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([50, 40], dtype=tf.float32), name='weight')
b2 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight')
b3 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

w4 = tf.compat.v1.Variable(tf.random.normal([40, 40], dtype=tf.float32), name='weight')
b4 = tf.compat.v1.Variable(tf.zeros([40], dtype=tf.float32), name='bias')
layer4 = tf.nn.softmax(tf.matmul(layer3, w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([40, 7], dtype=tf.float32), name='weight')
b5 = tf.compat.v1.Variable(tf.zeros([7], dtype=tf.float32), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(labels=y, logits=hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 4. 모델 훈련
epochs = 10000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train.tolist()})

        if step % 100 == 0:
            print("Step:", step, "Loss:", loss_val)

    # 훈련된 모델을 통해 예측값 출력
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})
    print("Predictions:", y_pred)

    # 평가 지표 계산
    y_pred_label = np.argmax(y_pred, axis=1)
    y_test_label = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_label, y_pred_label)
    print("Accuracy:", accuracy)



