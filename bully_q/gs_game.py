import sys
from contextlib import closing
from six import StringIO
from gym import utils
import discrete
import numpy as np


class GeneralSumEnv(discrete.DiscreteEnv):

    def __init__(self, payoff = ''):
        
        # state encoding
        # 0 = (0, 0) = (C, C); 
        # 1 = (0, 1) = (C, D); 
        # 2 = (1, 0) = (D, C); 
        # 3 = (1, 1) = (D, D)
        self.num_state = 1
        self.num_action = 3
        num_state = self.num_state
        num_action = self.num_action**2

        if payoff == '':
            #payoff = [[-1,-1], [-3,0], [0,-3], [-2,-2]]
            payoff = [[20, 10], [0, 0], [0, 0], 
                    [30, 0], [10, 20], [0, 0],
                    [0, 0], [0, 0], [15, 15]]

        self.payoff = payoff

        initial_state_distrib = np.ones(num_state) / num_state

        P = {state: {action: []
                     for action in range(num_action)} for state in range(num_state)}
        for state in range(num_state):
            for action in range(num_action):
                new_state = state
                reward = self.payoff[action]
                done = True
                P[state][action].append((1.0, new_state, reward, done))

        discrete.DiscreteEnv.__init__(
            self, num_state, num_action, P, initial_state_distrib)

