from .episodic_fifo import EPISODIC_FIFO
from madrl.nui.nui import NUI
from collections import deque
from random import randint
import random as random
from math import exp
import numpy as np

class NUI_ERM(EPISODIC_FIFO):
    """ Lenient Episodic Replay Memory implementation """

    def __init__(self, config):
        '''
        Sub-class of EPISODIC_FIFO. 
        Uses negative update intervals to determine which transitions should
        be stored in seperate ERM queues for meta actions.
        :param config: Replay Memory Hyperparameters
        '''
        super(NUI_ERM, self).__init__(config)
        self.__intervals = NUI(config) # Create Negative Update Intervals instance
        self.__ERMs = dict((i,deque([])) for i in range(config.meta_actions)) 
        self.__episodeCounter = 0

    @property
    def ERMs(self):
        return self.__ERMs

    @ERMs.setter
    def ERMs(self, value):
        raise Exception("Can't modify ERMs.")

    @ERMs.deleter
    def ERMs(self):
        raise Exception("Can't delete ERMs.")

    @property
    def episodeCounter(self):
        return self.__episodeCounter

    @episodeCounter.setter
    def episodeCounter(self, value):
        raise Exception("Can't modify episodeCounter.")

    @episodeCounter.deleter
    def episodeCounter(self):
        raise Exception("Can't delete episodeCounter.")

    @property
    def intervals(self):
        return self.__intervals

    @intervals.setter
    def temperatures(self, value):
        raise Exception("Can't modify intervals.")

    @intervals.deleter
    def temperatures(self):
        raise Exception("Can't delete intervals.")

    def delNonMax(self, reward, key):
        '''
        Removes entries replay memory below reward provided:
        :param float reward against which rewards insidte the RM are compared.
        :param int key: meta action key
        '''
        for i in range(len(self.__ERMs[key])):
            ep = self.__ERMs[key].popleft() 
            if not(ep[len(ep)-1][3] < reward):
                self.__ERMs[key].append(ep)

    def getSize(self):
        ''' 
        Returns the number of transitions currently stored inside the list. 
        '''
        count = 0
        for i in range(self.c.meta_actions):
            count += len(self.__ERMs[i]) 
        return count

    def add_experience(self, t):
        ''' 
        Method used to add state transitions to the 
        replay memory. 
        :param t(ransition): Tuple containing state transition tuple
        '''
        # Add transition to episode list:
        self.addStateTransition(t)
        if t[4] == 1: # If the transition is terminal
            if self.aboveThreshold():
                self.__intervals.decayLowerBound(t[5], t[3])     
            self.addToStrategyEpisodes(self._episode, t[5])
            self.clearEpisode() # Reset for next episode

    def addToStrategyEpisodes(self, ep, key):
        '''
        Adds episode to meta action queue.
        :param episode to add to queue
        :param key indicating which meta action was used
        '''
        if len(self.__ERMs[key]) >= self.c.nui.max_episodes:
            self.__ERMs[key].popleft() 
        self.__ERMs[key].append(ep)

    def clearEpisode(self):
        '''
        Method clears the transitions stored in the 
        episode list and increments the episode counter. 
        '''
        self._episode = [] # Reset for next episode
        self.__episodeCounter += 1

    def getLowerBound(self, key):
        '''
        returns l for a state s_t
        '''
        return self.__intervals.l[key]
 
    def aboveThreshold(self):
        '''
        :return bool: True if the number of transitions stored is above the learning threshold.
        '''
        return True if self.__episodeCounter > self.c.nui.decay_threshold else False

    def get_mini_batch(self):
        '''
        Method returns a mini-batch of sample traces.
        :return list traces: List of traces
        '''
        self._episodes = []
        for i in range(self.c.meta_actions):
            self._episodes = self._episodes + list(self.__ERMs[i])

        samples = [] # List used to store n traces used for sampling
        # Episodes are randomly choosen for sequence sampling:
        indexes = [random.randrange(len(self._episodes)) for i in range(self.c.erm.batch_size)]
        # From each of the episodes a sequence is selected:
        if len(self.c.dim) == 3:
            # From each of the episodes a sequence is selected:
            for i in indexes:
                samples.append(self._episodes[i][random.randint(0, len(self._episodes[i])-1)])
        else:
            for i in indexes:
                transition = random.randint(self.c.erm.sequence_len, len(self._episodes[i]))
                # State-trajectories are stored in lists.
                # Storing these each time provides memory to 
                # run multiple training runs in paralle at the 
                # cost of efficiency.
                o = []  
                o_tp1 = []
                for j in range(transition-self.c.erm.sequence_len, transition):
                    o.append(self._episodes[i][j][0])
                    o_tp1.append(self._episodes[i][j][1])
                transitionTuple = np.copy(self._episodes[i][transition-1])
                transitionTuple[0] = observations
                transitionTuple[1] = observations_tp1
                samples.append(transitionTuple)
        return samples


