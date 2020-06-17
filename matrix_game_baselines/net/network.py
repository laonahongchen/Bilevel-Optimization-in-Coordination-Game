import tensorflow as tf
import numpy as np
import tflearn

class Network(object):
    """ Abstract Network Class """

    def __init__(self, name, c):
        '''
        :param string name: Used for name scope
        :param config dict: c
        '''
        self.c = c
        self._name = name
        '''
        if len(c.dim) > 3 or len(c.dim) < 2:
            raise NotImplementedError('Invalid input dimensions.')
        '''
        if c.matrix_game:
            self.inputs = tf.placeholder('float32', shape[None, 2])
        elif c.use_conv == False:
            self.inputs = tf.placeholder('float32', shape=[None] + c.dim) 
        elif c.cnn.format == "NHWC":
            if len(c.dim) == 3:
                self.inputs = tf.placeholder('float32', shape=[None] + c.dim) 
            elif len(c.dim) == 2:
                self.inputs = tf.placeholder('float32', shape=[None, c.dim[0], c.dim[1], c.erm.sequence_len]) 
        elif c.cnn.format == "NCHW":
            if len(c.dim) == 3:
                self.inputs = tf.placeholder('float32', shape=[None, c.dim[2], c.dim[0], c.dim[1]]) 
            elif len(c.dim) == 2:
                self.inputs = tf.placeholder('float32', shape=[None, c.erm.sequence_len, c.dim[0], c.dim[1]]) 
            
        self.buildNetwork()
        self._fetch = {'outputs':self.outputs}


    def buildNetwork(self):
        '''
        Abstract Method buildNetwork
        '''
        raise NotImplementedError('Method buildNetwork not implemented.')

    def fetch(self, name, sess, observation):
        '''
        Method evaluates specified network output.
        :param string name: Name of output
        :param sess: TensorFlow session
        :param tensor observation: Network input
        :param tensor dropout: Used for dropout
        '''
        init = {self.inputs: observation}
        return self._fetch[name].eval(init, session=sess)


    def getWInit(self):
        w_init = tflearn.initializations.uniform(minval=-self.c.fcfe.init_min,\
                                                 maxval= self.c.fcfe.init_max)

    def fetchWithIndeces(self, name, sess, observation, indices):
        '''
        Method evaluates specified network output.
        :param string name: Name of output
        :param sess: TensorFlow session
        :param tensor observation: Network input
        :param tensor dropout: Used for dropout
        :param indeces: used to request specific q-values
        '''
        init = {self.inputs: observation, self.outputsIndices:indices}
        return self._fetchWithIndecies[name].eval(init, session=sess)


