from tensorflow.contrib.layers.python.layers import initializers
from exploration.epsilon_greedy import EpsilonGreedy
from net.feature_extractors import featureExtraction
from net.network import Network
import tensorflow as tf
from net.ops import hLoss
from drl.drl import DRL
import numpy as np
import tflearn

from saliency.saliency import Saliency

class DQN(DRL):
    """ DQN Implementation """

    def __init__(self, config):
        '''
        :param int agentID: Agent's ID
        :param dict config: Dictionary containing hyperparameters
        '''
        self.networks = [("cNet_", "tNet_")]
        super(DQN, self).__init__(config)
        self.getQTP1 = self.double if self.c.dqn.double == True else self.vanilla
        self.explore = EpsilonGreedy(self.cNet, config, self.sess)
        #self.saliency = Saliency(config, self.cNet, self.sess)

    def getAction(self, o_t, explore=True):
        '''
        Load action using observation as input
        :param tensor o_t: observation
        '''
        if len(self.c.dim) == 3:
            o_t = np.array(o_t).squeeze()
            self.current = np.copy(o_t)
        self.action = self.explore(o_t, self.episodeCounter, self.aboveLearningThreshold(), explore=explore)
        return self.action

    def getSaliencyCoordinates(self, obs, coordinates, location, hw):
        '''
        Used to get saliency coordinates for agent
        :param tensor: Observation
        :param vector: coordinates for which saliency is to be loaded
        :param vector: Agent location
        :param tuple: Height and width
        '''
        return self.saliency.coordinates(np.copy(obs), coordinates, location, hw)

    def getSaliency(self, obs):
        '''
        Used to save saliency for agent
        :param tensor: Observation
        '''
        return self.saliency.output(np.copy(obs))

    def vanilla(self, o_tp1):
        '''
        Method computes target for vanilla RDQN
        :param np.array o_tp1: Observation at time t+1
        :return float: max Q-Value for o_tp1
        '''
        return self.tNet.fetch('maxOutputs', self.sess, o_tp1)

    def double(self, o_tp1):
        ''' 
        Method computes target for double-DQN
        :param np.array o_tp1: State at time t+1
        :return float: Q-Value for o_tp1 from the target network
                       based on the arg max from the current network.
        '''
        predicted_actions = self.cNet.fetch('actions', self.sess, o_tp1)
        return self.tNet.fetchWithIndeces('outputsUsingIndices',\
                                          self.sess,\
                                          o_tp1,\
                                          [[idx, pred_a] for idx, pred_a in\
                                          enumerate(predicted_actions)])

    def getCurrentQ(self):
        '''
        Returns Q-Value for current observation action pair
        '''
        return self.cNet.fetch('outputs', self.sess, [self.current])[0][self.action]

    def calcTargets(self, terminal, o_tp1, reward):
        '''
        Returns target used for updating the network.
        :param terminal: 1 if final state transition of an episode, 0 otherwise
        :param tensor o_tp1: observation at time t plus 1
        :param float reward: reward received at time t plus 1
        :return tensor containing target valus
        '''
        # Target Q-Values for o_tp1 are obtained
        q_tp1 = self.getQTP1(o_tp1)
        # Tragets are calculated
        targets = (1.0 - np.array(terminal)) *\
                  self.c.gamma *\
                  q_tp1 +\
                  np.array(reward)
        # Upper bound is set for targets 
        targets[targets > self.c.dqn.max] = self.c.dqn.max
        #if sum(reward) > 0.0: 
        #    print(reward)
        #print(self.c.id)
        #if 1.0 in targets:
        #    print(targets)
        return targets

    def loadDict(self, targets, action, o_t):
        '''
        Preperation of optimisation dict used to train the network
        :param tensor targets: target values for network update
        :param tensor action: contains action used in each sample
        :param tensor o_t: observation at time t for each sample
        :return dict: used for initialising optimiser and cNet
        '''
        return {self.targets: targets,
                self.actions: action,
                self.cNet.inputs: o_t}

    def optUsingDict(self, optDict):
        '''
        Carry out optimsation
        :param optDict: used for initialisation
        '''
        _, delta, deltaAfter, actions, activeQValues, loss = self.sess.run([self.optim, self.delta, self.deltaAfter, self.actions, self.activeQValues, self.loss], optDict)
        return activeQValues


    def optimise(self):
        '''
        Optimises DQN
        '''
        o_t, o_tp1, action, reward, terminal = self.getUnzippedSamples()
        if self.c.cnn.format == "NHWC" and len(self.c.dim) == 2:
            o_t = np.moveaxis(o_t, 1, -1)
            o_tp1 = np.moveaxis(o_tp1, 1, -1)
        optDict = self.loadDict(self.calcTargets(terminal, o_tp1, reward), action, o_t)
        self.optUsingDict(optDict)

    def setOptVars(self):
        '''
        Set optimiser variables
        '''
        self.targets = tf.placeholder('float32', [None], name='targets')
        self.actions = tf.placeholder('int64', [None], name='actions')

    def setOptimiser(self):
        '''
        Optimiser used to train the DQN
        '''
        with tf.device(self.c.gpu):
            with self.g.as_default():
                with tf.variable_scope('optimiser'):
                    self.setOptVars()
                    actionsOneHot = tf.one_hot(self.actions,\
                                               self.c.outputs,\
                                               1.0,\
                                               0.0,\
                                               name='actionsOneHot')
                    self.activeQValues = tf.reduce_sum(self.cNet.outputs *\
                                                  actionsOneHot,\
                                                  reduction_indices=1,\
                                                  name='activeQValues')
                    self.delta = self.targets - self.activeQValues

                    self.deltaAfter = self.deltaProcessing()
                    with tf.name_scope('loss'):
                        self.loss = tf.reduce_mean(hLoss(self.deltaAfter), name='loss')                   
                        self.optim = tf.train.AdamOptimizer(self.c.dqn.alpha).minimize(self.loss)


    def addNetworks(self):
        '''
        Instantiates current and target networks.
        '''
        # Build current network
        self.cNet = self.Net("cNet_"+ self._name, self.c)
        # Build target network
        self.tNet = self.Net("tNet_"+ self._name, self.c)

    class Net(Network):

        """ Network used to approximate Q-Values """
        def __init__(self, name, c, K=14):
            '''
            :param string name: Used for name scope
            :param config dict: c
            '''
            self.K = K
            super(DQN.Net, self).__init__(name, c)
            self._fetch.update({'actions':self.actions})
            self._fetch.update({'maxOutputs':self.maxOutputs})
            self._fetch.update({'simHash':self.simHash})
            self._fetchWithIndecies = {'outputsUsingIndices':self.outputsUsingIndices}

        def buildNetwork(self):
            '''
            Build DQN
            '''
            with tf.device(self.c.gpu):
                with tf.variable_scope(self._name):
                
                    # Add Feature Extraction Layers
                    self.features = featureExtraction(self.inputs, self.c)
                    self.addOutputs()

        def addOutputs(self):
            '''
            Adds output layers to the graph
            '''
            # Outputs
            w_init = tflearn.initializations.xavier()
            self.outputs = tflearn.fully_connected(self.features, self.c.outputs, weights_init=w_init)
            self.maxOutputs = tf.reduce_max(self.outputs, axis=1)
            self.outputsIndices = tf.placeholder('int32', [None, None], 'outputsIndices')
            self.outputsUsingIndices = tf.gather_nd(self.outputs, self.outputsIndices)
            self.actions = tf.argmax(self.outputs, axis=1)
                
            # SimHash Add-on
            print(tf.shape(self.features))
            self.A = tf.get_variable('A', [self.K, 20], tf.float32,\
	    			     tf.random_normal_initializer(stddev=1.0))
            self.simHash = tf.sign(tf.matmul(self.A, self.features, transpose_b=True))


