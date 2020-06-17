from tensorflow.contrib.layers.python.layers import initializers
from .ops import linear, conv2d, conv2d_transpose, hLoss
from functools import reduce
from .network import Network
import tensorflow as tf
import numpy as np

class AUTOENCODER(Network):

    def __init__(self, config, name, k=16):
        self._format = config.cnn.format
        self._padding = config.cnn.p
        self._k = k
        self._learning_rate = config.dqn.alpha
        super(AUTOENCODER, self).__init__(name, config)
        self._fetch.update({'simHash':self.simHash})
        
    def buildNetwork(self):
        with tf.device(self.c.gpu):
            initializer = initializers.xavier_initializer()
            activation_fn = tf.nn.relu
            # current network
            with tf.variable_scope(self._name):
                # used for SimHash [k, D]
                self.A = tf.get_variable('A', [self._k, 512], tf.float32, tf.random_normal_initializer(stddev=1.0))

                # Inputs are normalised
                self.l0 = tf.div(self.inputs, self.c.cnn.max_in)

                self.l1 = conv2d(self.l0, 32, [4, 4], [2, 2],\
                          self._format, self._padding, name='l1')
                self.l2 = conv2d(self.l1, 64, [2, 2], [1, 1], \
                          self._format, self._padding, name='l2')
	
                shape = self.l2.get_shape().as_list()
                self.l2_flat = tf.reshape(self.l2, [-1, reduce(lambda x, y: x * y, shape[1:])])

                self.l3 = linear(self.l2_flat, 1024, activation=activation_fn, name='l3')

                # Autoencoder:
                self.dense_sig = linear(self.l3, 512, activation=tf.nn.sigmoid, name='dense_sig')

                self.dense_sig_with_noise = self.dense_sig + tf.random_normal(shape=tf.shape(self.dense_sig), mean=0.0, stddev=0.3, dtype=tf.float32)
               
                self.dl1_flat = linear(self.dense_sig_with_noise, 1024, activation=activation_fn, name='dl1_flat')

                self.dl1_2d = tf.reshape(self.dl1_flat, [32, 4, 4, 64])
                
                self.l1_decode = conv2d_transpose(self.dl1_2d,
                                                  [2, 2],
                                                  64,
                                                  32,
                                                  [8,8], 
                                                  [1, 2, 2, 1],
                                                  initializer,
                                                  activation_fn, 
                                                  name='l1_decode')

                self.outputs = conv2d_transpose(self.l1_decode,
                                                  [4,4],
                                                  32,
                                                  1,
                                                  self.c.dim, 
                                                  [1, 2,2, 1],
                                                  initializer,
                                                  activation_fn, 
                                                  name='l2_decode')
                self.ae_loss = tf.nn.l2_loss(hLoss(self.l0 - self.outputs))
                self.optim = tf.train.AdamOptimizer(self._learning_rate).minimize(self.ae_loss)
                self.simHash = tf.sign(tf.matmul(self.A, self.dense_sig, transpose_b=True))

