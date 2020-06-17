from erm.episodic_fifo_lenient import EPISODIC_FIFO_LENIENT
from exploration.epsilon_greedy import EpsilonGreedy
from exploration.tgreedy import TGreedy
from net.autoencoder import AUTOENCODER
from drl.dqn.dqn import DQN
import tensorflow as tf
import numpy as np

class Leniency(DQN):
    """ LDQN Implementation """

    def __init__(self, config):
        '''
        :param int agentID: Agent's ID
        :param dict config: Dictionary containing hyperparameters
        '''
        super(Leniency, self).__init__(config)
        self.index = None
        self._index_tp1 = None
        if config.dqn.exploration== 'epsGreedy':
            self.explore = EpsilonGreedy(self.cNet, config, self.sess)
        elif config.dqn.exploration== 'tBarGreedy':
            self.explore = TGreedy(self.cNet, self.replay_memory, config, self.sess)

    def getAction(self, o_t, explore=True):
        '''
        Load action using observation as input
        :param tensor o_t: state
        '''
        if len(self.c.dim) == 3:
            o_t = np.array(o_t).squeeze()
            self.current = np.copy(o_t)
        if self.c.dqn.exploration== 'tBarGreedy':
            self.action = self.explore(o_t,\
                                       self.index,\
                                       explore=explore)
        elif self.c.dqn.exploration== 'epsGreedy':
            self.action = self.explore(o_t,\
                                       self.episodeCounter,\
                                       self.aboveLearningThreshold(),\
                                       explore=explore)
        return self.action

    def replayMemoryInit(self):
        '''
        Instantiate Replay Memory
        '''
        if self.c.leniency.hashing == 'AutoEncoder':
            self.replay_memory = EPISODIC_FIFO_LENIENT(self.c, self.ae, self.sess)
        else:  
            self.replay_memory = EPISODIC_FIFO_LENIENT(self.c, self.cNet, self.sess)

    def addAuxNets(self):
        '''
        Method for adding Auxilliary (or just additional) networks to the graph. 
        '''
        if self.c.leniency.hashing == 'AutoEncoder':
            self.ae = AUTOENCODER(self.c, "ae_" + self._name)

    def optimise(self):
        '''
        Optimises LDQN
        :param int t: Timestep used to determine if sync should take place
        '''
        o_t, o_tp1, action, reward, terminal,_,_, leniency, _ = self.getUnzippedSamples()
        o_t = np.array(o_t)
        o_tp1 = np.array(o_tp1)
        o_t = o_t.reshape(o_t.shape[0], 1)
        o_tp1 = o_tp1.reshape(o_tp1.shape[0], 1)
        #print(o_t.shape, o_tp1.shape)
        if self.c.cnn.format == "NHWC" and len(self.c.dim) == 2:
            o_t = np.moveaxis(o_t, 1, -1)
            o_tp1 = np.moveaxis(o_tp1, 1, -1)
        optDict = self.loadDict(self.calcTargets(terminal, o_tp1, reward), action, o_t)
        optDict.update({self.leniency:leniency})
        self.optUsingDict(optDict)

    def setOptVars(self):
        '''
        Set optimiser variables
        '''
        self.leniency = tf.placeholder('float32', [None], name='leniency')
        self.targets = tf.placeholder('float32', [None], name='targets')
        self.actions = tf.placeholder('int64', [None], name='actions')

    def deltaProcessing(self):
        '''
        Method applies leniency to losses. However, below
        leniency is no longer used to determin whehter a 
        negative same should be included or not, but rather
        to scale the impact of negative deltas.
        :param vector delta: vector containing losses
        :param vector leniency: vector containing leniency values
        :return self.delta: Returns modified delta value
        '''
        return tf.where(tf.greater(self.delta, tf.constant(0.0)),\
                         self.delta, self.delta*(1.0-self.leniency))


    def storeTransitionTuple(self, reward, terminal, new_state, reduced_observation):
        '''
        Add tuple to replay memory:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        '''
        if self.index == None:
            self.index = self.replay_memory.getHashKey(self.current)

        if reduced_observation is None:
            self._index_tp1 = self.replay_memory.getHashKey(self.current)
        else:
            self._index_tp1 = self.replay_memory.getHashKey(reduced_observation)

        self.replay_memory.add_experience([np.copy(self.current),\
                                           np.copy(new_state),\
                                           self.action,\
                                           reward,\
                                           terminal,
                                           self.index,
                                           self._index_tp1])
  
        self.index = self._index_tp1

    def feedback(self, reward, terminal, new_state, reduced_observation=None):
        '''
        Agent is provided with feedback:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        '''
        if terminal == 1:
            self.episodeCounter += 1
        
        self.storeTransitionTuple(reward, terminal, new_state, reduced_observation)

