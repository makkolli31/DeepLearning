import tensorflow as tf

tf.set_random_seed(1)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.truncated_normal([784, 30]))
b1 = tf.Variable(tf.truncated_normal([1, 30]))
W2 = tf.Variable(tf.truncated_normal([30, 10]))
b2 = tf.Variable(tf.truncated_normal([1, 10]))


def sigma(x):
    return 1. / (1. + tf.exp(-x))


def sigmaprime(x):
    return sigma(x) * (1. - sigma(x))

l1 = tf.matmul(X, W1) + b1
a1 = sigma(l1)
l2 = tf.matmul(a1, W2) + b2
y_pred = sigma(l2)

assert y_pred.shape.as_list() == Y.shape.as_list()
diff = (y_pred - Y) # dE/dy_pred

# dE/dl2 = dE/dy_pred * dy_pred/dl2
d_l2 = diff * sigmaprime(l2)

# b2 !!!!
# dE/db2 = dE/dl2 * dl2/db2 = dE/dl2 * 1 = dE/dy_pred * dy_pred/dl2
d_b2 = d_l2

# w2 !!!!
# dE/dw2 = dE/do2 * do2/dw2 = dE/dl2 * dl2/do2 * do2/dw2 = dE/dy_pred * dy_pred/dl2 * 1 * do2/dw2 = (dE/dy_pred * dy_pred/dl2) * X
d_w2 = tf.matmul(tf.transpose(a1), d_l2)

# dE/da1 = dE/do2 * do2/da1 = dE/dl2 * dl2/do2 * do2/da1 = dE/dy_pred * dy_pred/dl2 * 1 * do2/da1 = (dE/dy_pred * dy_pred/dl2) * W
d_a1 = tf.matmul(d_l2, tf.transpose(W2))

# dE/dl1 = dE/da1 * da1/dl1
d_l1 = d_a1 * sigmaprime(l1)

# b1 !!!!
# dE/db1 = dE/dl1 * dl1/db1 = dE/dl1 * 1
d_b1 = d_l1

# w1 !!!!
# dE/dw1 = dE/do1 * do1/dw1 = dE/dl1 * dl1/do1 * X = dE/dl1 * 1 * X
d_w1 = tf.matmul(tf.transpose(X), d_l1)

learning_rate = 0.5
step = [
    tf.assign(W1, W1 - learning_rate * d_w1),
    tf.assign(b1, b1 - learning_rate * tf.reduce_mean(d_b1, 0)),#reduction_indices=[0])),

    tf.assign(W2, W2 - learning_rate * d_w2),
    tf.assign(b2, b2 - learning_rate * tf.reduce_mean(d_b2, 0)),#reduction_indices=[0])),
]

acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

#cost = tf.multiply(diff, diff)
#cost = tf.reduce_mean(diff*diff)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict={X:batch_xs, Y:batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict={X: mnist.test.images[:1000], Y: mnist.test.labels[:1000]})
        print(res)

#cost = diff * diff ??
