import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1. 데이터

x,y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)
# print(y[:10]) #[151.  75. 141. 206. 135.  97. 138.  63. 110. 310.]

y = y.reshape(-1, 1) #(442,)에서 y가 (442, 1)로 바뀜

#(442, 10) * (10, 1) + b(?) = (442, 1) -> w : (10, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=337)


# Convert the input data to float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# print(x_train.shape, y_train.shape) #(353, 10) (353, 1)
# print(x_test.shape, y_test.shape) #(89, 10) (89, 1)

x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
w = tf.compat.v1.Variable(tf.random_normal([10, 1], dtype=tf.float32), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

#2. 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

#3. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

train = optimizer.minimize(loss)


#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Training
epochs = 1000
for epoch in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss_val)

# Evaluation
y_train_pred = sess.run(hypothesis, feed_dict={x: x_train})
y_test_pred = sess.run(hypothesis, feed_dict={x: x_test})

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("R2 Score (Train):", r2_train)
print("R2 Score (Test):", r2_test)

# Close the session
sess.close()