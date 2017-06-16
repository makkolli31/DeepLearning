import tensorflow as tf
import numpy as np

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0,1,0,2,3,3]]
x_one_hot = [[[1,0,0,0,0], # h
            [0,1,0,0,0], # i
            [1,0,0,0,0], # h
            [0,0,1,0,0], # e
            [0,0,0,1,0], # l
            [0,0,0,1,0]]] # l

y_data = [[1,0,2,3,3,4]]

num_classes = 5 # 출력 Y의 선택지
input_dim = 5 # one hot size ( num_classes와 동일 할 듯 )
hidden_size = 5 # Layer Output
batch_size = 1 # One Sentence
sequence_length = 6 # 문자 수
#rnn_size = len(char_dic)
#batch_size = 1
#output_size = 4

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y Label

cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
fc_b = tf.get_variable("fc_b", [num_classes])
outputs = tf.matmul(X_for_fc, fc_w) + fc_b
#outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=Y)
total_loss= tf.reduce_mean(loss)

#weights = tf.ones([batch_size, sequence_length])
#sequence_loss = tf.contrib.seq2seq.sequence_loss( logits=outputs, targets=Y, weights=weights)
#loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(total_loss)



prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str: ", ''.join(result_str))