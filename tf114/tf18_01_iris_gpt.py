import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
if  tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(337)

def load_iris_data():
    # Load iris dataset
    iris = load_iris()

    # Split the dataset into features and labels
    x = iris.data
    y = iris.target.reshape(-1, 1)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)

    # Convert the input data to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return x_train, x_test, y_train, y_test

def build_model():
    xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
    yp = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])

    w = tf.compat.v1.Variable(tf.random.normal([4, 3], dtype=tf.float32), name='weight')
    b = tf.compat.v1.Variable(tf.zeros([1, 3], dtype=tf.float32), name='bias')

    logits = tf.matmul(xp, w) + b
    hypothesis = tf.nn.softmax(logits)

    # Define loss function and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(yp), logits=logits))
    train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    return xp, yp, hypothesis, loss, train

def train_model(x_train, y_train, xp, yp, hypothesis, loss, train, sess):
    epochs = 500

    for epoch in range(epochs):
        loss_val, _ = sess.run([loss, train], feed_dict={xp: x_train, yp: y_train})

        if epoch % 10 == 0:
            print(epoch, 'loss:', loss_val)

def evaluate_model(x_test, y_test, xp, hypothesis, sess):
    y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={xp: x_test})
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

def main():
    x_train, x_test, y_train, y_test = load_iris_data()
    xp, yp, hypothesis, loss, train = build_model()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        train_model(x_train, y_train, xp, yp, hypothesis, loss, train, sess)
        evaluate_model(x_test, y_test, xp, hypothesis, sess)

if __name__ == '__main__':
    main()
