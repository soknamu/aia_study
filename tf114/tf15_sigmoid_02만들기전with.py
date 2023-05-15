import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
tf.compat.v1.set_random_seed(337)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] #(6,2)
y_data = [[0],[0],[0],[1],[1],[1]]                  #(6,1)

x = tf.compat.v1.placeholder(tf.float32, shape= [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape= [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name= 'weight', dtype=tf.float32) #shape를 맞춰주기 위해 [3,1]
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name= 'bias', dtype=tf.float32)

hypothesis = tf.compat.v1.matmul(x, w)+b

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)

#3-2 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 15000
    for epoch in range(epochs):
        _, loss_val,w_val, b_val = sess.run([train, loss, w, b], 
                    feed_dict={x : x_data, y : y_data})
        if epoch % 10 ==0:
            print('epochs :', epoch, 'loss: ',loss_val)
                
sess = tf.compat.v1.Session()
# #4. 평가, 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
# y_predict = x_test * w_val + b_val -> 넘파이랑 텐서랑 행렬곱을 했더니 에러발생, 그래서 matmul 사용해야됨.
y_predict = tf.matmul(x_test, w_val) + b_val

y_aaa=sess.run(y_predict, feed_dict = {x_test : x_data})

r2 = r2_score(y_aaa, y_data)
print('r2 : ', r2)

mse = mean_squared_error(y_aaa, y_data)
print('mse : ', mse)

sess.close()
