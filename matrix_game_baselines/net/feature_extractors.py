from tflearn.activations import relu
from functools import reduce
import tensorflow as tf
from .ops import conv2d
import tflearn

def featureExtraction(i, c):
    '''
    Returns layers that can be used for feature extraction
    :param vector: inputs i
    :param dict: config dictionary c
    :return: Add features layers to tf graph
    '''
    return convLayers(i, c) if c.use_conv == True else fcLayers(i, c)

def recurrentFeatureExtraction(i, c, batch_size, sequence_length, cell, state):
    '''
    Returns layers that can be used for feature extraction with recurrence
    :param vector: inputs i
    :param dict: config dictionary c
    :return: Adds features layers and recurrent cell to tf graph
    '''
    features = convLayers(i, c) if c.use_conv == True else fcLayers(i, c)
    features = tf.reshape(features, [batch_size, sequence_length, c.recurrent.h_size])
    output, state = tf.nn.dynamic_rnn(cell, features, initial_state=state, dtype=tf.float32)
    return tf.reshape(output, [-1, c.recurrent.h_size]), state

def convLayers(inputs, c):
    '''
    Conv layers that can be used for feature extraction
    :param vector: inputs
    :param dict: config dictionary c
    :return: Add conv features layers to tf graph
    '''
    weights_init = tflearn.initializations.xavier()
    bias_init = tf.constant_initializer(0.1)
    #print(inputs.get_shape())
    inputs = tf.div(inputs, c.cnn.max_in)
    layer = 0
    for (o, k, s) in zip(c.cnn.outdim, c.cnn.kernels, c.cnn.stride):
        inputs = conv2d(inputs, o, [k, k], [s, s], name='conv' + str(layer), format=c.cnn.format)
        layer += 1                
    shape = inputs.get_shape().as_list()
    inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x * y, shape[1:])])
    return relu(tflearn.fully_connected(inputs, c.cnn.fc, weights_init=weights_init,bias_init=bias_init))

def fcLayers(inputs, c):
    '''
    Fully connected layers that can be used for feature extraction
    :param vector: inputs
    :param dict: config dictionary c
    :return: Add fully connected features layers to tf graph
    '''
    inputs = tf.layers.flatten(inputs)
    if c.fcfe.normalise:
         inputs = tf.div(tf.subtract(inputs,tf.reduce_min(inputs)),\
			 tf.subtract(tf.reduce_max(inputs), tf.reduce_min(inputs)))
    for s in c.fcfe.layers:
        inputs = relu(tflearn.fully_connected(inputs, s, weights_init=c.fcfe.w_init))
    return inputs
