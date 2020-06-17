import random as random
import numpy as np
class EpsilonGreedy:

    def __init__(self, net, config, sess):
        self.__c = config
        self.__net = net
        self.__sess = sess
        self.__ep = 0
        self.__explore_step = 500
        self.reset()

    def __call__(self, o_t, episode, aboveLearningThreshold, explore=True):
        '''
        Implementation of epsilon greedy action selection strategy.
        On policy action is selected with probability 1 - epsilon.
        :param tensor o_t: Observation
        :return int: action selected
        '''
        self.update(episode, aboveLearningThreshold)
        #explore = False
        with self.__sess.as_default():
            if random.random() < self.__epsilon and explore or episode < self.__explore_step:
                return random.randrange(self.__c.outputs)
            elif not explore and random.random() < 0.2:
                return random.randrange(self.__c.outputs)
            else:
                return self.__net.fetch('actions', self.__sess, [o_t])[0]

    def update(self, episode, aboveLearningThreshold):
        '''
        Method used to update epsilon value, which 
        determins how greedy the e-greedy exploration 
        strategy is.
        '''
        #if aboveLearningThreshold and episode > self.__ep:
        self.__epsilon = max((self.__epsilon * self.__c.epsgreedy.discount), self.__c.epsgreedy.min)
        '''
        if episode > self.__ep:
            self.__ep += 1
        '''
    def reset(self):
        '''
        Initialises self.episilon for epsilon greedy exploration.
        '''
        self.__epsilon = self.__c.epsgreedy.initial

