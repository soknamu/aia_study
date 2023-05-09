import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.set_random_seed(123)

# 1. 데이터
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 2. 모델 구성
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.08424)
train = optimizer.minimize(loss)

# 3-2. 훈련
#히스토리
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) #이게 변수로.
    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x: [1, 2, 3, 4, 5], y: [2, 4, 6, 8, 10]})
        if step % 1 == 0:
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
    x_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    y_pred = x_test * w_val + b_val
    print('[6, 7, 8] 예측:', sess.run(hypothesis, feed_dict={x: x_data}))
    
print(loss_val_list)
print(w_val_list)
######################  [실습]  ############################
# import matplotlib.pyplot as plt
# # plt.plot(loss_val_list)
# # plt.xlabel('epochs')
# # plt.ylabel('loss')
# # plt.show()

# # plt.plot(w_val_list)
# # plt.xlabel('epochs')
# # plt.ylabel('weights')
# # plt.show()

# plt.plot(w_val_list, loss_val_list)
# plt.xlabel('weight')
# plt.ylabel('loss')
# plt.show()
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 첫 번째 서브플롯: loss_val_list 그래프
axs[0].plot(loss_val_list)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('loss')

# 두 번째 서브플롯: w_val_list 그래프
axs[1].plot(w_val_list)
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('weights')

# 세 번째 서브플롯: w_val_list vs. loss_val_list 그래프
axs[2].plot(w_val_list, loss_val_list)
axs[2].set_xlabel('weight')
axs[2].set_ylabel('loss')

plt.tight_layout()  # 서브플롯 간의 간격 조절
plt.show()

