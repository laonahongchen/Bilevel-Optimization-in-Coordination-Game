from net.feature_extractors import featureExtraction
from net.network import Network
import tensorflow as tf
import tflearn

class ACTOR(Network):
    """ Actor Network """

    def __init__(self, name, c): 
        '''
        :param string name: Used for name scope
        :param config dict: c
        '''
        super(ACTOR, self).__init__(name, c)
        self._fetch.update({'scaled_outputs':self.scaled_outputs})

    def fetch(self, name, sess, observations):
        '''
        Method evaluates specified network output.
        :param string name: Name of output
        :param sess: TensorFlow session
        :param tensor observations: Network input
        '''
        if self.c.ddpg.prev_action_nodes is not None:
            init = {self.inputs: observations[:][0],\
                    self.previous_actions:observations[:][1]}
        else:
            init = {self.inputs: observations}
        return self._fetch[name].eval(init, session=sess)

    def getWInit(self):
        w_init = tflearn.initializations.uniform(minval=-self.c.ddpg.w_init,\
                                                 maxval= self.c.ddpg.w_init)

    def buildNetwork(self):
        '''
        Build Actor
        '''
        with tf.device(self.c.gpu):
            with tf.variable_scope(self._name):
                self.features = featureExtraction(self.inputs, self.c)
                if self.c.ddpg.prev_action_nodes is not None:
                    self.previous_actions = tf.placeholder(tf.float32, shape=[None] + [self.c.outputs]) 
                    term1 = tflearn.fully_connected(self.features, self.c.ddpg.t1_nodes)
                    term2 = tflearn.fully_connected(self.previous_actions,  self.c.ddpg.prev_action_nodes)
                    self.features = tflearn.activation(tf.matmul(self.features, term1.W) +\
				                       tf.matmul(self.actions, term2.W)\
					               + term2.b, activation='relu')
                self.outputs = tflearn.fully_connected(self.features,\
                                                       self.c.outputs,\
                                                       activation='tanh',\
                                                       weights_init=self.getWInit())
                self.scaled_outputs = tf.multiply(self.outputs, self.c.ddpg.upper_bound) 
