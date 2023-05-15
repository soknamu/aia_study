# multi valuable

import tensorflow as tf
from sklearn.metrics import r2_score
tf.compat.v1.set_random_seed(123)

#1. data
            # st nd rd th th
x1_data = [ 73., 93., 89., 96., 73.]
x2_data = [ 80., 88., 91., 98., 66.]
x3_data = [ 75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

#[실습] 맹그러봐!

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # w 3개, b 1개
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

#2. model

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. compile
loss = tf.reduce_mean(tf.square(hypothesis - y))  #mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.00001)
train = optimizer.minimize(loss)

# #3-2. 훈련
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     epochs = 2000
#     for step in range(epochs):
#         _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b], 
#                 feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y : y_data})

#         if step % 10 == 0:
#             print(step, loss_val, w1_val, w2_val, w3_val, b_val)
#     x_test_data = [93., 88., 89., 90., 70.]
#     y_test_data = [185., 180., 182., 179., 150.]
    
#     y_predict_val = sess.run(hypothesis, feed_dict={x1: x_test_data, x2: x_test_data, x3: x_test_data, b: b_val})
#     r2 = r2_score(y_test_data, y_predict_val)

#     print("R2 Score:", r2)   

#tensor의 연산형식은 자료형.
# sess = tf.compat.v1.Session()
# epochs = 2001
# for step in range(epochs):
#     cost_val, hy_val, _  = sess.run([ loss, hypothesis ,train], 
#                          feed_dict= {x1: x1_data, x2: x2_data, x3: x3_data, y : y_data})

#     if step % 10 == 0:
#         print(epochs, 'loss : ', cost_val)

#     # Prediction
#     y_pred = sess.run(hypothesis, feed_dict={x: x_test})

#     # Evaluation
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)

#     print('r2 :', r2)
#     print('mae :', mae)

# sess.close()