from tensorflow.contrib.layers.python.layers import initializers
from scipy.misc import imsave 
import tensorflow as tf
import numpy as np
import config


# To be removed. tf.losses.huber_loss to be used instead
def hLoss(x, delta=1.0): 
    '''
    Returns Huber Loss for vector X.
    :param vector x: Vector of losses
    :param float delta: Determines when small or large 
                        residuals are returned
    :return vector: Huber Loss
    '''
    residual = tf.abs(x)
    condition = tf.less(residual, delta)
    small = 0.5 * tf.square(residual)
    large = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small, large)


def conv2d(input, output_dimension, kernel, stride,
           format='NCHW', padding='SAME', name='conv2d',
           activation=tf.nn.relu,  
           initializer=tf.contrib.layers.xavier_initializer()):
    '''
    Add conv2d layer to network.
    :param tensor input: Input to be processed by layer.
    :param tensor output_dimension: Dimensions of layer output.
    :param list kernel: kernel dimensions
    :param list stride: stride values
    :param activation: TF activation function or set to None
    :param format: NHWC/NCHW (Batch, height, width and channels) 
    :param string padding: Type of padding to be used
    :param string name: Layer name
    :param initializer: Variable initializer
    :return tensor output
    '''
    print(format)
    with tf.variable_scope(name) as scope:
        if format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel[0], 
                            kernel[1], 
                            input.get_shape()[1], 
                            output_dimension]
        elif format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape= [kernel[0], 
                           kernel[1], 
                           input.get_shape()[-1], 
                           output_dimension]
        # Weights are innitialised based upon kernel size and output dimensions:
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        b = tf.get_variable('biases', [output_dimension], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, w, stride, padding, data_format=format)
        output = tf.nn.bias_add(conv, b, format)
    # Output is returned
    return activation(output) if activation != None else output

def conv2d_transpose(input,
                     kernel_size,
                     input_channels,
                     output_channels,
                     heightAndWidth,                     
                     strides,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     activation=tf.nn.relu, 
                     name='conv2d_transpose'):
    '''
    Up-sampling 2-D Layer (deconvolutoinal Layer)
    :param tensor input: Input to be processed by layer.
    :param int input_channels: Number of input channels
    :param int output_channels: Number of output channels
    :param list output_shape: Output height and width
    :param list stride: stride values
    :param initializer: Variable initializer
    :param activation: TF activation function or set to None
    :param string name: Layer name
    :return tensor output
    '''
    with tf.variable_scope(name) as scope:
        # Size of the kernel to convolve over the transposed (padded) layer
        kernel = [kernel_size[0], kernel_size[1], output_channels, input_channels]
    
        # Weights are innitialised based upon kernel size:
        w = tf.get_variable('w', kernel, tf.float32, initializer=initializer)
        b = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.1))
    
        # Output shape is defined:
        output_shape = [32, heightAndWidth[0], heightAndWidth[1], output_channels]      
   
        # conv2d_transpose is used to perfrom a form of "deconvolution" (acutually it's just convolution 
        # on a padded tensor...)
        output = tf.nn.conv2d_transpose(input, w, output_shape=output_shape, strides=strides, padding='SAME') + b
  
    # Output is returned
    return activation(output) if activation != None else output

def conv3d(input, output_shape, is_train, k_h=5, k_w=5, k_d=5, stddev=0.02, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, k_d, input.get_shape()[-1], output_shape],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input, w, strides=[1, 2, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, is_training=is_train,updates_collections=None)
    return bn

def linear(input, output_size, activation=None, name='linear',\
           initializer=tf.contrib.layers.xavier_initializer()):
    '''
    Add linear layer to network.
    :param tensor input: Input to be processed by layer.
    :param tensor output_dimension: Dimensions of layer output.
    :param activation: TF activation function or set to None
    :param string name: Layer name
    :param initializer: Variable initializer
    :return tensor output
    '''
    shape = input.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, initializer=initializer)
        b = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        output = tf.nn.bias_add(tf.matmul(input, w), b)
    # Output is returned
    return activation(output) if activation != None else output

