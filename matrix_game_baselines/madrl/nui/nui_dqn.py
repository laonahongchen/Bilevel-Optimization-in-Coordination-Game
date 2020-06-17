from collections import deque
from erm.nui_erm import NUI_ERM
import tensorflow as tf
import random as random
from drl.dqn.dqn import DQN
import numpy as np

class NUI_DQN(DQN):
    """ Strategic-LDQN Implementation """

    def __init__(self, config):
        '''
        :param int agentID: Agent's ID
        :param dict config: Dictionary containing hyperparameters
        '''
        super(NUI_DQN, self).__init__(config)
        # Dict used to establish average qValues for each meta action:
        self.qValues = dict((i,deque([])) for i in range(config.meta_actions))  
        # Dict used to store max rewards received for each meta action:
        self.maxRewards = dict.fromkeys(range(config.meta_actions), -1.0)
        # Error margin for max reward:
        self.__eps = config.nui.eps

    def replayMemoryInit(self):
        '''
        Instantiate Replay Memory
        '''
        from erm.nui_erm import NUI_ERM as ReplayMemory
        self.replay_memory = ReplayMemory(self.c)

    def aboveLearningThreshold(self):
        '''
        Returns true if transitions stored in ERM is above the learning threshold.
        '''
        return self.replay_memory.getSize() >= self.c.nui.threshold

    def optimise(self):
        '''
        Optimises NUI DQN
        '''
        o_t, o_tp1, a, r, t, idx = self.getUnzippedSamples()
        optDict = self.loadDict(self.calcTargets(t, o_tp1, r), a, o_t)
        qVals = self.optUsingDict(optDict)
        self.storeQValues(qVals, idx, t)

    def storeQValues(self, qValues, idx, t):
        '''
        Stores qValues in queue 
        :param vector: Q-Value vector
        :param vector: index vector
        :param vector: terminal vector
        '''
        for i in range(len(idx)):
            if t[i] > 0:
                if len(self.qValues[idx[i]]) > 49: 
                    self.qValues[idx[i]].popleft()
                self.qValues[idx[i]].append(qValues[i])

    def storeTransitionTuple(self, reward, terminal, new_state, meta_action):
        '''
        Add tuple to replay memory:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        :param integer index: index of meta action
        '''
        clear = False
        if terminal > 0.0:
            print("RM Size: " + str(self.replay_memory.getSize()))
            print(str(self.c.id) + " meta_action: " + str(meta_action)) 
            self.replay_memory.intervals.trackRewards(meta_action, reward)
            if reward > self.maxRewards[meta_action]:
                self.maxRewards[meta_action] = reward
                self.replay_memory.delNonMax(reward, meta_action)
            l = self.replay_memory.getLowerBound(meta_action)
            if self.maxRewards[meta_action] - self.__eps  > reward and l - 1.0 - self.__eps > reward:
                self.replay_memory.clearEpisode()
                clear = True
        if clear == False:
            self.replay_memory.add_experience([np.copy(self.current),\
                                               np.copy(new_state),\
                                               self.action,\
                                               reward,\
                                               terminal,\
                                               meta_action])

    def feedback(self, reward, terminal, new_state, meta_action=None):
        '''
        Agent is provided with feedback:
        :param float reward: Reward received after transition
        :param int terminal: 1 if terminal state, 0 otherwise
        :param np.array new_state: state entered  at time tp1
        '''
        if terminal == 1:
            self.episodeCounter += 1
        
        self.storeTransitionTuple(reward, terminal, new_state, meta_action)

