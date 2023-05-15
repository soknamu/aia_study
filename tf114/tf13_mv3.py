#행렬연산
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
tf.compat.v1.set_random_seed(337)

#1. 데이터
x_data = [[73, 51, 65],                     #(5, 3)
          [92, 98, 11], 
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],[182], [180], [205], [142]] #(5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3]) #-> x의 값은 계속 변경할수 있기때문에
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]), name= 'weight') #shape를 맞춰주기 위해 [3,1]
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([]), name= 'bias')

#2. 모델
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b #행렬의 곱 
# -> 두개가 크게 상관없는데 오류가 뜨면 밑에꺼 권장.

# x.shape = (5,3), y.shape = (5,1)
# hy = x * w + b
#    =(5, 3) * (3,1) = (5,1)
#hypothesis 는 y_data와 shape가 같음

#3. 컴파일, 훈련\
#3-1. compile
loss = tf.reduce_mean(tf.square(hypothesis - y))  #mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.000001)
train = optimizer.minimize(loss)
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# print(sess.run([hypothesis, w, b], feed_dict = {x : x_data}))

# #3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 5000
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train ,loss, w, b], feed_dict={x: x_data, y: y_data})
        if step % 10 == 0:
            print(step, loss_val, w_val)
    # 예측값 계산
    y_pred_val = sess.run(hypothesis, feed_dict={x: x_data})

    # TensorFlow 텐서를 NumPy 배열로 변환
    y_data_np = np.array(y_data)
    y_pred_np = np.array(y_pred_val)

    # R2 스코어 계산
    r2 = r2_score(y_data_np, y_pred_np)

    print("R2 Score:", r2)

