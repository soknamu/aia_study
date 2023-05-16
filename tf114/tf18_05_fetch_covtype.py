import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(337)

# 1. 데이터
x, y = fetch_covtype(return_X_y=True)
print(x.shape, y.shape)  # (581012, 54) (581012,)

# One-Hot Encoding
y_onehot = pd.get_dummies(y).values
print(y_onehot.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, train_size=0.8, shuffle=True, random_state=337)

# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w = tf.compat.v1.Variable(tf.random.normal([54, 7], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1, 7], dtype=tf.float32), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x_place, w) + b)

# 3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y_place * tf.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 500

for epoch in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x_place: x_train, y_place: y_train})

    if epoch % 10 == 0:
        print(epoch, 'loss :', loss_val)

# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x_place: x_test})
y_aaa = np.argmax(y_pred, axis=1)
y_test_label = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_label, y_aaa)
print('acc :', acc)

sess.close()
