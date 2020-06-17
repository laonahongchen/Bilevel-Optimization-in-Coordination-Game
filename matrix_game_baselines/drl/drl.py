from collections import deque
import random as random
import tensorflow as tf
import numpy as np

class DRL(object):

    """ Deep Reinforcement Learner """
    def __init__(self, c):
        '''
        :param dict: Dictionary containing hyperparameters
        '''
        self.c = c # Hyperparamter dict
        self._name = "agent" + str(c.id)
        self.t = 0
        self.episodeCounter = 0
        with tf.device(self.c.gpu):
            self.g = tf.Graph()
            with self.g.as_default():
                self.addNetworks()
                self.addAuxNets()
                if self.c.optimise:
                    self.setOptimiser()
                self.initParameters()
                self.createNetworkSyncOps()
                self.sess.run(self.syncNetworks)
        self.replayMemoryInit()
        self.current = np.zeros(self.c.dim, dtype=np.float32)
        # tau is used to store state transition trjectory:
        self.__tau = deque([np.zeros(self.c.dim, dtype=np.float32)\
                            for i in range(c.erm.sequence_len)],\
                            c.erm.sequence_len)
           
    def replayMemoryInit(self):
        '''
        Instantiate Replay Memory
        '''
        if self.c.erm.type == 'FIFO':
            from erm.fifo import FIFO as ReplayMemory
        elif self.c.erm.type == 'EPISODIC_FIFO':
            from erm.episodic_fifo import EPISODIC_FIFO as ReplayMemory
        elif self.c.erm.type == 'NUI_ERM':
            from erm.nui_erm import NUI_ERM as ReplayMemory
        self.replay_memory = ReplayMemory(self.c)

    def getUnzippedSamples(self):
        '''
        :return unzipped samples obtained from replay memory
        '''
        # Samples are obtained from the replay memory and unzipped
        samples = self.replay_memory.get_mini_batch()
        return zip(*samples)

    def addAuxNets(self):
        '''
        Abstract method for adding Auxilliary (or additional) networks. 
        '''
        pass

    def addNetworks(self):
        '''
        Method used to build and initiate agent's
        current and target networks.
        '''
        raise NotImplementedError('Method addNetworks not implemented.')

    def optimise(self, t):
        '''
        Abstract method train
        '''
        raise NotImplementedError('Method optimise not implemented.')
    
    def getAction(self, s_t, explore=True):
        '''
        Load action using state (observation) as input
        :param tensor s_t: state
        :param bool explore: Set to false for greedy action selection
        '''             
        raise NotImplementedError('Method getAction not implemented.')

    def setOptimiser(self, name):
        '''
        Abstract method setOptimizer
        '''
        raise NotImplementedError('Method setOptimizer not implemented.')

    def initParameters(self):
        '''
        Used to initialise parameters
        '''
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,\
                                gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(tf.variables_initializer(set(tf.global_variables())))

        # Saver object
        self.saver = tf.train.Saver(max_to_keep=10000)

    def createNetworkSyncOps(self):
        ''' 
        Method used to sync current and target networks
        '''
        assigns = [] # List used to store assign operations
        incAssigns = [] # List used to store inc assign operations
        for cn, tn in self.networks:
            with self.g.as_default():
                v = tf.trainable_variables()
                # Current network vars loaded into cn_vars
                cn_vars = filter(lambda x: x.name.startswith(cn + self._name), v)
                # Target network vars loaded into tn_vars
                tn_vars = filter(lambda x: x.name.startswith(tn + self._name), v)
                for t, c in zip(tn_vars, cn_vars):
                    assigns.append(t.assign(c.value()))
                    incAssigns.append(t.assign(tf.multiply(c.value(), self.c.tau)\
                                   + tf.multiply(t.value(), 1. - self.c.tau))) 
                self.syncNetworks = tf.group(*assigns)
                self.incSyncNetworks = tf.group(*incAssigns)

    def syncTargetNetworks(self):
        '''
        Synchronises current and target networks
        '''
        if self.c.inc_sync:
            self.sess.run(self.incSyncNetworks)
        elif self.t%self.c.sync_time == 0:
            self.sess.run(self.syncNetworks)

    def saveModel(self, folder, step):
         '''
         Save Model
         '''
         self.saver.save(self.sess, folder + self._name + "/" + self._name, global_step=step)


    def restoreModel(self, folder, model): 
        '''
        Restore Model
        '''
        with self.g.as_default():
            self.saver = tf.train.import_meta_graph(folder+model + '.meta')
            self.saver.restore(self.sess, folder+model)

    def deltaProcessing(self):
        '''
        Method can be overriden to do something useful with 
        the delta values.
        :return self.delta: Currently returns unmodified delta value
        '''
        return self.delta

    def opt(self):
        '''
        Opt network.
        '''
        # Check if optimisation operations should be performed
        
        if self.aboveLearningThreshold() and self.t % self.c.erm.train_steps == 0:      
            self.optimise()
            self.syncTargetNetworks()

    def move(self, o_t, explore=True):
        '''
        Method returns move to be made by the agent
        based upon selected exploration strategy.
        :param numpy array o_t: containing observation
        :return action (what the action "is" may 
                        depends on the DRL used)
        '''

        # Increment timestep
        self.t += 1

        # Return action
        self.current = np.copy(o_t)
        self.__tau.append(np.copy(o_t))
        if self.c.cnn.format == "NHWC" and len(self.c.dim) == 2:
            return self.getAction(np.moveaxis(self.__tau, 0, -1), explore)        
        return self.getAction(self.__tau, explore)

    def aboveLearningThreshold(self):
        '''
        Returns true if transitions stored in ERM is above the learning threshold.
        '''
        #return True 
        #print(self.replay_memory.getSize())
        return self.replay_memory.getSize() >= self.c.erm.threshold

    def storeTransitionTuple(self, reward, terminal, new_state):
        '''
        Add tuple to replay memory:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        '''
        self.replay_memory.add_experience([np.copy(self.current),\
                                                np.copy(new_state),\
                                                self.action,\
                                                reward,\
                                                terminal])

    def feedback(self, reward, terminal, new_state):
        '''
        Agent is provided with feedback:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        '''
        if terminal == 1:
            self.episodeCounter += 1
        
        self.storeTransitionTuple(reward, terminal, new_state)

