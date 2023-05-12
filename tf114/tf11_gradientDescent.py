import tensorflow as tf

x_train = [1, 2, 3] #[1]
y_train = [1, 2, 3] #[2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name= 'weight')

hypothesis = x * w
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse



############################## optimizer ######################################
lr = 0.1

# gradient = tf.reduce_mean((w * x -y) * x) 틀림
gradient = tf.reduce_mean((x * w -y) * x) #  w = w-lr * 미분e/ 미분w 이랑 연관이 있음. / 이게 gradient의 식(미분값)
# gradient = tf.reduce_mean((hypothesis -y) * x) #-> 이거랑 같음

descent = w - lr * gradient #gradient가 미분e/ 미분w임. ===로스의 변환값.
update = w.assign(descent) #  w = w -lr * gradient -> 계속 기울기 값을 업데이트.
#편미분 : 내가 미분할 값만 미분, 나머지를 상수로 봄.
#미분에 미분은 미분미분이다.
################################################################################

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v, w_v  = sess.run([update, loss, w], feed_dict= {x : x_train, y : y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

sess.close()

print("============= W history ===============")
print(w_history)
print("============= Loss history ===============")
print(loss_history)