#################[실습] R2, mse 맹그러!!!!!!##########################
import tensorflow as tf
from sklearn.metrics import r2_score,mean_absolute_error

x_train = [1, 2, 3] #[1]
y_train = [1, 2, 3] #[2]
x_test = [4,5,6]
y_test = [4,5,6]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name= 'weight')

hypothesis = x * w
loss = tf.reduce_mean(tf.square(hypothesis - y)) #mse

############################## optimizer ######################################
lr = 0.1 

# gradient = tf.reduce_mean((w * x -y) * x) 틀림
gradient = tf.reduce_mean((x * w -y) * x) #  w = w-lr * 미분e/ 미분w 이랑 연관이 있음. / 이게 gradient의 식(미분값)

descent = w - lr * gradient
update = w.assign(descent)

w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(31):
    _, loss_v, w_v  = sess.run([update, loss, w], feed_dict= {x : x_train, y : y_train})
    print(step, '\t', loss_v, '\t', w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)

    # Prediction
    y_pred = sess.run(hypothesis, feed_dict={x: x_test})

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('r2 :', r2)
    print('mae :', mae)

sess.close()

# r2 : 0.999999989276489
# mae : 8.344650268554688e-05

#y_predict = x_test * w_v