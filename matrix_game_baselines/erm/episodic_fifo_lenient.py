from madrl.leniency.temperature import Temperature
from .episodic_fifo import EPISODIC_FIFO
from collections import deque
import random as random
from math import exp
import numpy as np
import xxhash

class EPISODIC_FIFO_LENIENT(EPISODIC_FIFO):
    """ Lenient Episodic Replay Memory implementation """

    def __init__(self, config, net, sess):
        '''
        Sub-class of EPISODIC_FIFO. Stores the amount 
        of leniency that should be applied when sampling
        a state transition.
        :param config: Replay Memory Hyperparameters
        '''
        super(EPISODIC_FIFO_LENIENT, self).__init__(config)
        self._t = Temperature(config, config.outputs, net, sess, self) # Create temperature instance
        self._tmc = config.leniency.tmc # Temperature Moderation Coeficient
        self._leniency_threshold = config.leniency.threshold 

    def getAvgTempUsingIndex(self, index):
        '''
        Gets avgerge temperature for a state based on index.
        :param int index: Hash key for a state.
        :return float: average temperature for the state belongin to index.
        '''
        return self._t.getAvgTempUsingIndex(index)

    def add_experience(self, transition):
        ''' 
        Method used to add state transitions to the replay memory. 
        :param transition: Tuple containing state transition tuple
        '''
        # Add transition to episode list:
        self.addStateTransition(transition)
        '''
        self._num_transitions_stored += 1
        while self.isFull():
            deletedEpisode = self._episodes.popleft() # Pop first entry if RM is full
            self._num_transitions_stored -= len(deletedEpisode)
        '''
        if transition[4] == 1: # If the transition is terminal
            self._num_transitions_stored += len(self._episode)
            while self.isFull():
                deletedEpisode = self._episodes.popleft() # Pop first entry if RM is full
                self._num_transitions_stored -= len(deletedEpisode)
            self._episodes.append(self._episode) # Store episode
            self._episode = [] # Reset for next episode

    def getSize(self):
        ''' 
        Returns the number of transitions currently stored inside the list. 
        '''
        return self._num_transitions_stored

    def aboveLeniencyThreshold(self):
        '''
        :return bool: True if the number of transitions stored
                       is above the learning threshold.
        '''
        return True if self._num_transitions_stored > self._leniency_threshold else False

    def addStateTransition(self, transition):
        '''
        Adds state transition to self._episodes
        :param tuple transition: transition to be added
        '''
        _, _, action, _, terminal, _, _ = transition
        if transition[4] > 0.0:
            print("key1: " +str(transition[5]))
        temperature = self._t.getTemperatureUsingIndex(transition[5], action)
        leniency = 1 - exp(-self._tmc * temperature)
        transition.append(leniency)
        transition.append(temperature)
        self._episode.append(transition)
        if transition[4] == 1.0:
            print("Terminal Leniency: " + str(leniency))
            print("Max Leniency: " + str(1 - exp(-self._tmc * self._t.getMaxTemperature())))
            self._t.incEps()
        # If RM is full and terminal transition has been reached
        if transition[3] >= 0.0 and transition[4] == 1.0 and self.aboveLeniencyThreshold():
            # Update temperatures for state action pairs visited
            self._t.updateTemperatures(self._episode) 

    def getHashKey(self, o_t):
        '''
        Loads hash-key for observation o_t
        :param tensor o_t: Observation for which key is required.
        :return int: hash key for observation o_t
        '''
        return self._t.getHash(o_t) 

    def getMaxTemperature(self):
        '''
        :return float: Max temperature from temperature instance
        '''
        return self._t.getMaxTemperature()        
