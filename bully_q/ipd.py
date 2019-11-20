import sys
from contextlib import closing
from six import StringIO
from gym import utils
import bully_q.discrete as discrete
import numpy as np


class IpdEnv(discrete.DiscreteEnv):
   
    def __init__(self, payoff = ''):
        
        # 0 = (0, 0) = (C, C); 
        # 1 = (0, 1) = (C, D); 
        # 2 = (1, 0) = (D, C); 
        # 3 = (1, 1) = (D, D)
        self.num_state = 4
        self.num_action = 2
        num_state = 4
        num_action = 4

        if payoff == '':
            payoff = [[-1,-1], [-3,0], [0,-3], [-2,-2]]
            # payoff = [[3, 3], [1, 1], [1, 1], [2, 2]]
            #payoff = [[1, 0], [0, 1], [0, 1], [1, 0]]

        self.payoff = payoff

        initial_state_distrib = np.array([0.25, 0.25, 0.25, 0.25])

        P = {state: {action: []
                     for action in range(num_action)} for state in range(num_state)}
        for state in range(num_state):
            for action in range(num_action):
                new_state = action
                reward = [[-1,-1], [-3,0], [0,-3], [-2,-2]][action]
                done = False
                P[state][action].append((1.0, new_state, reward, done))

        discrete.DiscreteEnv.__init__(
            self, num_state, num_action, P, initial_state_distrib)