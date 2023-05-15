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

# print(x_train.shape, y_train.shape) #(353, 10) (353, 1)
# print(x_test.shape, y_test.shape) #(89, 10) (89, 1)

x_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y_place = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1], name='weight')) #정규분포의 의한 랜덤값 10개가 들어감.
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], name='bias'))

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

epochs = 101
for epochs in range(epochs):
     _, loss_val, w_val, b_val = sess.run([train ,loss, w, b], feed_dict={x: x_place, y: y_place})
     if epochs % 10 == 0:
        print(epochs, loss_val, w_val)
    # 예측값 계산
     y_pred = sess.run(hypothesis, feed_dict={x: x_place})

    # TensorFlow 텐서를 NumPy 배열로 변환
     y_data_np = np.array(y_place)
     y_pred_np = np.array(y_pred)

    # R2 스코어 계산
     r2 = r2_score(y_data_np, y_pred_np)

     print("R2 Score:", r2)