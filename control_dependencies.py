import tensorflow as tf

def my_aaa(inputs):
    my_concat = tf.concat([inputs, [0]], 0)
    assign_op = tf.assign(inputs, my_concat, validate_shape=False)
    with tf.control_dependencies([my_concat]):
        return tf.Print(inputs, [inputs])


x = tf.Variable([], dtype=tf.int32, validate_shape=False, trainable=False)
#concat = tf.concat([x, [0]], 0)
#asdf = tf.Print(concat, [concat])
asdf = my_aaa(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        sess.run([asdf])

'''
# I define a "shape-able" Variable
x = tf.Variable([], dtype=tf.int32, validate_shape=False, trainable=False)
# I build a new shape and assign it to x
concat = tf.concat([x, [0]], 0)
assign_op = tf.assign(x, concat, validate_shape=False)

with tf.control_dependencies([assign_op]):
  # I print x after the assignment
  # Note that the Print call is on "x" and NOT "assign_op"
    print_op_dep = tf.Print(x, data=[x], message="print_op_dep:")

    #new_x = x.read_value()
    #print_op_dep = tf.Print(new_x, data=[new_x], message="print_op_dep:")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(3):
    sess.run(print_op_dep)
'''