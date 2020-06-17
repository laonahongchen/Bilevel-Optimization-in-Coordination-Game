from net.feature_extractors import featureExtraction
from net.network import Network
from copy import deepcopy
import tensorflow as tf
import tflearn

class CRITIC(Network):
    """ Build Critic Network """

    def __init__(self, name, c):
        '''
        :param string name: Used for name scope
        :param config dict: c
        '''
        super(CRITIC, self).__init__(name, c)
        self._fetch.update({'action_grads':self.action_grads})

    def fetch(self, name, sess, observations, actions):
        '''
        Method evaluates specified network output.
        :param string name: Name of output
        :param sess: TensorFlow session
        :param tensor observations: Network input
        :param tensor actions: Network input
        '''
        if self.c.ddpg.prev_action_nodes is not None:
            init = {self.inputs: observations[:][0],\
                    self.previous_actions:observations[:][1],\
                    self.actions: actions}
        else:
            init = {self.inputs: observations, self.actions: actions}
        return self._fetch[name].eval(init, session=sess)

    def getWInit(self):
        w_init = tflearn.initializations.uniform(minval=-self.c.ddpg.w_init,\
                                                 maxval= self.c.ddpg.w_init)
    def buildNetwork(self):
        '''
        Build Critic
        '''
        with tf.device(self.c.gpu):
            with tf.variable_scope(self._name):
                # Placeholders
                self.actions = tf.placeholder(tf.float32, shape=[None] + [self.c.outputs]) 
                
                # Feature Layers
                cpy = deepcopy(self.c)
                cpy.fcfe.layers = [400]
                self.features = featureExtraction(self.inputs, cpy)
                term1 = tflearn.fully_connected(self.features, self.c.ddpg.t1_nodes)
                term2 = tflearn.fully_connected(self.actions,  self.c.ddpg.t2_nodes)
                if self.c.ddpg.prev_action_nodes is not None:
                    self.previous_actions = tf.placeholder(tf.float32, shape=[None] + [self.c.outputs]) 
                    term3 = tflearn.fully_connected(self.previous_actions,  self.c.ddpg.prev_action_nodes)
                    self.features = tflearn.activation(tf.matmul(self.features, term1.W) +\
				                       tf.matmul(self.actions, term2.W)\
					               + term2.b +\
				                       tf.matmul(self.previous_actions, term3.W)\
					               + term3.b, activation='relu')
                else:
                    self.features = tflearn.activation(tf.matmul(self.features, term1.W) +\
				                       tf.matmul(self.actions, term2.W)\
					               + term2.b, activation='relu')
  
                # Outputs:
                self.outputs = tflearn.fully_connected(self.features, 1, weights_init=self.getWInit())

                # Action gradients
                self.action_grads = tf.gradients(self.outputs, self.actions)[0]
