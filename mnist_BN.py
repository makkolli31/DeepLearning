#
# https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#

import numpy as np, tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

epsilon = 1e-3

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

##-------------------------------------------------------
w1 = tf.Variable(w1_initial)
b1 = tf.Variable(tf.zeros([100]))
z1 = tf.matmul(x,w1)+b1
l1 = tf.nn.sigmoid(z1)
##-------------------------------------------------------

##-------------------------------------------------------
w1_BN = tf.Variable(w1_initial)
z1_BN = tf.matmul(x,w1_BN)

batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])

z1_hat = (z1_BN - batch_mean1) / tf.sqrt(batch_var1 + epsilon)

scale1 = tf.Variable(tf.ones([100]))
beta1 = tf.Variable(tf.zeros([100]))

BN1 = scale1 * z1_hat + beta1
l1_BN = tf.nn.sigmoid(BN1)
##-------------------------------------------------------

##-------------------------------------------------------
w2 = tf.Variable(w2_initial)
b2 = tf.Variable(tf.zeros([100]))
z2 = tf.matmul(l1,w2)+b2
l2 = tf.nn.sigmoid(z2)
##-------------------------------------------------------

##-------------------------------------------------------
w2_BN = tf.Variable(w2_initial)
z2_BN = tf.matmul(l1_BN,w2_BN)
batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
scale2 = tf.Variable(tf.ones([100]))
beta2 = tf.Variable(tf.zeros([100]))
BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
l2_BN = tf.nn.sigmoid(BN2)
##-------------------------------------------------------

w3 = tf.Variable(w3_initial)
b3 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(l2,w3)+b3)

w3_BN = tf.Variable(w3_initial)
b3_BN = tf.Variable(tf.zeros([10]))
y_BN = tf.nn.softmax(tf.matmul(l2_BN,w3_BN)+b3_BN)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_BN)

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_prediction_BN = tf.equal(tf.arg_max(y_BN,1) ,tf.arg_max(y_,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_prediction_BN, tf.float32))

zs, BNs, acc, acc_BN = np.empty((0,100)), np.empty((0,100)), [], []

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in tqdm.tqdm(range(40000)):
    batch = mnist.train.next_batch(60)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})
    train_step_BN.run(feed_dict={x:batch[0], y_:batch[1]})
    if i % 50 is 0 :
        res = sess.run([accuracy, accuracy_BN, z2, BN2], feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        acc = np.append(acc, res[0])
        acc_BN = np.append(acc_BN, res[1])

        zs = np.append(zs, [np.mean(res[2], axis=0)], axis=0)
        BNs = np.append(BNs, [np.mean(res[3], axis=0)], axis=0)

zs, BNs, acc, acc_BN = np.array(zs), np.array(BNs), np.array(acc), np.array(acc_BN)

"""
fig, ax = plt.subplots()

ax.plot(range(0, len(acc) * 50, 50), acc, label='Without BN')
ax.plot(range(0, len(acc) * 50, 50), acc_BN, label='With BN')
ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.8, 1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.show()
"""

"""
fig, axes = plt.subplots(5, 2, figsize=(6,12))
fig.tight_layout()

for i, ax in enumerate(axes):
    ax[0].set_title("Without BN")
    ax[1].set_title("With BN")
    ax[0].plot(zs[:,i])
    ax[1].plot(BNs[:,i])

plt.show()
"""

"""
predictions = []
correct = 0
for i in range(100):
    pred, corr = sess.run([tf.arg_max(y_BN,1), accuracy_BN],
                          feed_dict={x:[mnist.test.images[i]], y_:[mnist.test.labels[i]]})

    correct += corr
    predictions.append(pred[0])
print("PREDICTIONS:", predictions)
print("ACCURACY:", correct/100)
"""

