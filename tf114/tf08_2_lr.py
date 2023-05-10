import tensorflow as tf

tf.compat.v1.set_random_seed(123)

# 1. 데이터
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 2. 모델 구성
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) #이게 변수로.
    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: [1, 2, 3, 4, 5], y: [2, 4, 6, 8, 10]})
        if step % 10 == 0:
            print(step, loss_val, w_val, b_val)

    x_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    y_pred = x_test * w_val + b_val
    print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))
######################  [실습]  ############################



# x_data = [6,7,8]
#예측값을 뽑아라.
