import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape)  # (569, 30) (569,)

y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([30, 1], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x_place, w) + b)
epsilon = 1e-7  # epsilon 값을 조정하여 로스 값이 고정되는 문제를 해결

# 3. 컴파일
loss = tf.reduce_mean(y_place * tf.log(hypothesis + epsilon) + (1 - y_place) * tf.log(1 - hypothesis + epsilon))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 3000
for epoch in range(epochs):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_place: x_train, y_place: y_train})
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)

# 4. 평가, 예측
x_test_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y_predict = tf.sigmoid(tf.matmul(x_test_place, w_val) + b_val)
y_predict = tf.cast(y_predict > 0.5, dtype=tf.float32)

y_aaa = sess.run(y_predict, feed_dict={x_test_place: x_test})

acc = accuracy_score(y_aaa, y_test)
print('acc : ', acc)

mse = mean_squared_error(y_aaa, y_test)
print('mse : ', mse)

sess.close()
