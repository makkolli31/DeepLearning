import tensorflow as tf
import numpy as np

tf.set_random_seed(1)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
X_data = xy[:, 0:-1]
N = X_data.shape[0]
Y_data = xy[:, [-1]]
print(np.unique(Y_data))

print("Shape of X data", X_data.shape)
print("Shape of Y data", Y_data.shape)

nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

target = tf.one_hot(Y, nb_classes)
target = tf.reshape(target, [-1, nb_classes])
target = tf.cast(target, tf.float32)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

def sigma(x):
    return 1. / (1. + tf.exp(-x))


def sigmaprime(x):
    return sigma(x) * (1. - sigma(x))

layer_1 = tf.matmul(X, W) + b
y_pred = sigma(layer_1)

loss_i = -target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

assert y_pred.shape.as_list() == target.shape.as_list()

d_loss = (y_pred - target) / (y_pred * (1. - y_pred))# + 1e-7)
d_sigma = sigmaprime(layer_1)
d_layer = d_loss * d_sigma
d_b = d_layer
d_W = tf.matmul(tf.transpose(X), d_layer)

learning_rate = 0.01
train_step = [
    tf.assign(W, W - learning_rate * d_W),
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b))
]

prediction = tf.argmax(y_pred, 1)
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target,1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(target, feed_dict={Y:Y_data}))
    print(sess.run(y_pred, feed_dict={X: X_data}))
    print(sess.run(prediction, feed_dict={X: X_data}))
    print(sess.run(acct_mat, feed_dict={X: X_data, Y:Y_data}))
    print(sess.run(acct_res, feed_dict={X: X_data, Y:Y_data}))

    for step in range(500):
        sess.run(train_step, feed_dict={X:X_data, Y:Y_data})

        if step % 10 == 0:
            step_loss, acc = sess.run([loss, acct_res], feed_dict={X:X_data, Y:Y_data})
            print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}".format(step, step_loss, acc))

    pred = sess.run(prediction, feed_dict={X:X_data})
    for p, y in zip(pred, Y_data) :
        msg = "[{}]\t Prediction : {:d}\t True y: {:d}"
        print(msg.format(p == int(y[0]), p, int(y[0])))