import numpy
from collections import deque
class NUI():
    '''
    Class maintains lower bound interval values for meta actions
    (sequences of actions). Meta actions are
    identified via keys. The number of meta actions required
    is specified within the config file. 
    '''

    def __init__(self, config):
        '''
        :param dict: config, contains hyperparameters
        '''
        # Dict used to store lower boundary for each meta action:
        self.__l = dict.fromkeys(range(config.meta_actions), 2.0)
        # Dict used to store max reward received for each meta action:
        self.__maxRe = dict.fromkeys(range(config.meta_actions), -1.0)
        # Decay rate for temperatures:
        self.__decay = config.nui.decay
        # Error margin for max reward:
        self.__eps = config.nui.eps
        # Reward queue for computing mean and std
        self.__rewards = dict((i,deque([], 200)) for i in range(config.meta_actions)) 

    @property
    def eps(self):
        return self.__eps

    @eps.setter
    def eps(self, value):
        raise Exception("Can't modify eps.")

    @eps.deleter
    def eps(self):
        raise Exception("Can't delete eps.")

    @property
    def l(self):
        return self.__l

    @l.setter
    def l(self, value):
        raise Exception("Can't modify l.")

    @l.deleter
    def l(self):
        raise Exception("Can't delete l.")

    @property
    def maxRe(self):
        return self.__maxRe

    @maxRe.setter
    def maxRe(self, value):
        raise Exception("Can't modify maxRe.")

    @maxRe.deleter
    def maxRe(self):
        raise Exception("Can't delete maxRe.")

    @property
    def decay(self):
        return self.__decay

    @decay.setter
    def decay(self, value):
        raise Exception("Can't modify decay.")

    @decay.deleter
    def decay(self):
        raise Exception("Can't delete decay.")

    def trackRewards(self, key, reward):
        '''
        Method keeps track of actual rewards received for each key:
        :param generic: Key representing the meta action
        :param float: reward received after outcome occured
        '''
        self.__rewards[key].append(reward)

    def decayLowerBound(self, key, reward):
        '''
        Method decays temperature based on the key and reward:
        :param generic: Key representing the meta action
        :param float: reward received after outcome occured
        '''
        if key in self.__l:
            if reward > self.__maxRe[key]:
                self.__maxRe.update({key:reward})
            # Only decay if reward is max
            if reward >= self.__maxRe[key] - self.__eps:
                avg = numpy.array(list(self.__rewards[key])).mean()
                std = numpy.array(list(self.__rewards[key])).std(ddof=1)
                self.__l[key] = max(self.__l[key]*self.__decay, (avg-std-0.3)+1.0) 
                print(str(key) + ": " + str(self.__l[key]))
        else:
            raise ValueError('Abstract action key does not exist.')

