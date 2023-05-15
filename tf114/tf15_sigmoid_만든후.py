import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_squared_error
tf.compat.v1.set_random_seed(337)

#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]   # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                     # (6, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32)

#2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+ b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))     # mse
loss = tf.reduce_mean(y*tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis)) # = 'binary_crossentropy'

optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 25000
    
for step in range(epochs):
    cost_val, _ , w_val, b_val= sess.run([loss, train, w, b], 
                                feed_dict={x: x_data, y: y_data})
        
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)

#4. 평가, 예측            
x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
# y_predict = x_data * w_val + b_val    # 넘파이랑 텐서랑 행렬곱했더니 에러남, 그래서 아래 matmul 사용해야 됨
y_predict = tf.sigmoid(tf.matmul(x_test, w_val)) + b_val
# y_predict = tf.cast(y_predict> 0,5, dtype=tf.float32) 띄어쓰기 해야됨.
y_predict = tf.cast(y_predict > 0.5, dtype=tf.float32)

y_aaa = sess.run(y_predict, feed_dict={x_test:x_data})

# print(y_aaa)
# [[0.86980325]
#  [0.9538312 ]
#  [0.8634029 ]
#  [0.9769021 ]
#  [0.98375046]
#  [0.97561634]]
#print(type(y_aaa))  # <class 'numpy.ndarray'>

acc = accuracy_score(y_aaa, y_data)
print('r2 : ', acc)

mse = mean_squared_error(y_aaa, y_data)
print('mse : ', mse)

sess.close()