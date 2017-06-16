import tensorflow as tf

tf.set_random_seed(1)

x_data = [        [1, 2],        [2, 3],        [3, 1],        [4, 3],        [5, 3],        [6, 2]    ]

y_data = [        [1, 0],        [1, 0],        [1, 0],        [0, 1],        [0, 1],        [0, 1]    ]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 2])

#W1 = tf.Variable(tf.random_normal([2, 2]))
#b1 = tf.Variable(tf.random_normal([2]))
#L1 = tf.nn.sigmoid(tf.matmul(X, W1)+ b1)

#W2 = tf.Variable(tf.random_normal([2, 2]))
#b2 = tf.Variable(tf.random_normal([2]))
#L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([2, 2]))
b3 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.nn.softmax(tf.matmul(X, W3) + b3)

#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
cost = -tf.reduce_mean(Y * tf.log(hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    #h = sess.run([hypothesis], feed_dict={X: x_data, Y: y_data})
    #c = sess.run([predicted], feed_dict={X: x_data, Y: y_data})
    #a = sess.run([accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: ", a)
