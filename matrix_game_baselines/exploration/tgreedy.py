import random as random

class TGreedy:
    ''' Average Temperature Greedy Exploration '''

    def __init__(self, net, erm, config, sess):
        self.__config = config
        self.__net = net
        self.__sess = sess
        self.__erm = erm

    def __call__(self, s_t, idx, explore=True):
        '''
        Implementation of TBar-greedy action selection strategy.
        On policy action is selected with probability 1 - the
        average temperature value for s_t.
        :param tensor s_t: Observation
        :param tensor s_t: Observation
        :param bool explore: Greedy action is returned when set to false
        :return int: action selected
        '''
        if self.__erm.aboveLeniencyThreshold():
            temperature = self.__erm.getAvgTempUsingIndex(idx)
        else:
            temperature = 1.0
        with self.__sess.as_default():
            if random.random() < temperature**self.__config.leniency.ase and explore:
                return random.randrange(self.__config.outputs)
            else:
                return self.__net.fetch('actions', self.__sess, [s_t])[0]

    def update(self, episode, aboveLearningThreshold):
        '''
        Method used to update epsilon value, which 
        determins how greedy the e-greedy exploration 
        strategy is.
        '''
        pass

    def reset(self):
        '''
        Re-Init
        '''
        pass 
