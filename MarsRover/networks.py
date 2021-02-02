
import tensorflow as tf
import tensorflow.contrib.layers as c_layers



def rnn(input, n_units, n_classes):

    layer = {'weights': tf.Variable(tf.random_normal([n_units, n_classes])),
             'bias': tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell = tf.contrib.rnn.LSTMCell(n_units)

    initial_state = lstm_cell.zero_state(tf.shape(input)[0],dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, input, initial_state=initial_state, dtype=tf.float32)
    outputs=tf.reshape(outputs, [-1, n_units])

    outputs = tf.matmul(outputs, layer['weights']) + layer['bias']

    return outputs





