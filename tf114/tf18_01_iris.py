import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

tf.compat.v1.set_random_seed(337)

# 1. 데이터
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)  # (150, 4) (150,)

y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y_place = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([4, 3], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1, 3], dtype=tf.float32), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x_place, w) + b)

# 3-1 컴파일
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(y_place), logits=hypothesis))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 500

for epoch in range(epochs):
    loss_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x_place: x_train, y_place: y_train})

    if epoch % 10 == 0:
        print(epoch, 'loss :', loss_val)

# 4. 평가, 예측
x_test_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y_predict = tf.matmul(x_test_place, w_val) + b_val
y_aaa = np.argmax(sess.run(y_predict, feed_dict={x_test_place: x_test}), axis=1)

acc = accuracy_score(y_test, y_aaa)
print('acc : ', acc)

sess.close()
