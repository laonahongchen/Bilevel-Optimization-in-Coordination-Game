import sys
from contextlib import closing
from six import StringIO

# from gym import utils

import bilevel_pg.bilevelpg.environment.discrete_game as discrete
import numpy as np
from bilevel_pg.bilevelpg.environment.base_game import BaseGame
# import utils
import bilevel_pg.bilevelpg.environment.utils as utils


class GridEnv(discrete.DiscreteEnv):
    """
        Has the following members
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)
        (*) dictionary dict of dicts of lists, where
          P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS
        """



    def __init__(self):

        self.num_state = 81
        self.num_action = 4
        num_state = self.num_state
        num_action = self.num_action ** 2
        self.agent_num = 2
        self.action_num = 4
        self.game_name = 'grid_world'

        payoff = [[-1, -1] for i in range(num_state)]

        self.source0 = self.encode_pos(0, 2)
        self.source1 = self.encode_pos(2, 2)
        # self.target0 = self.encode_pos(1, 1)
        # self.target1 = self.encode_pos(1, 1)
        # self.target0 = self.encode_pos(0, 0)
        # self.target1 = self.encode_pos(2, 0)
        self.target0 = self.encode_pos(1, 0)
        self.target1 = self.encode_pos(1, 0)

        for i in range(9):
            state = self.encode_state(self.target0, i)
            payoff[state][0] = 10
            state = self.encode_state(i, self.target1)
            payoff[state][1] = 10

        self.payoff = payoff

        initial_state_distrib = np.zeros(num_state)
        initial_state_distrib[self.encode_state(self.source0, self.source1)] = 1

        P = {state: {action: []
                     for action in range(num_action)} for state in range(num_state)}
        for state in range(num_state):
            for action in range(num_action):
                [action0, action1] = utils.decode_action(action, self.num_action)
                new_state = self.next_state(state, action0, action1)
                reward = self.payoff[new_state]
                [pos0, pos1] = self.decode_state(new_state)
                # if new_state == self.encode_state(self.target0, self.target1):
                if pos0 == self.target0 or pos1 == self.target1:
                    done = True
                else:
                    done = False
                P[state][action].append((1.0, new_state, reward, np.array([done] * 2)))

        discrete.DiscreteEnv.__init__(
            self, num_state, num_action, P, initial_state_distrib)

        print('start state', np.array([[self.encode_state(self.source0, self.source1)], [self.encode_state(self.source0, self.source1)]]))

    def encode_state(self, pos0, pos1):
        return pos0 * 9 + pos1

    def decode_state(self, state):
        return [state // 9, state % 9]

    def encode_pos(self, x, y):
        return x + y * 3

    def decode_pos(self, pos):
        return [pos % 3, pos // 3]

    def out(self, x, y):
        if x < 0 or x > 2 or y < 0 or y > 2:
            return True
        else:
            return False

    def next_state(self, state, action0, action1):
        dir_x = [0, 0, -1, 1]  # up, down, left, right
        dir_y = [1, -1, 0, 0]  # up, down, left, right
        [pos0, pos1] = self.decode_state(state)

        [x0, y0] = self.decode_pos(pos0)
        new_x0 = x0 + dir_x[action0]
        new_y0 = y0 + dir_y[action0]
        if self.out(new_x0, new_y0):
            new_x0 = x0
            new_y0 = y0
        new_pos0 = self.encode_pos(new_x0, new_y0)
        if (pos0 == self.source0 and action0 == 1):  # hit the wall
            new_pos0 = pos0

        [x1, y1] = self.decode_pos(pos1)
        new_y1 = y1 + dir_y[action1]
        new_x1 = x1 + dir_x[action1]
        if self.out(new_x1, new_y1):
            new_x1 = x1
            new_y1 = y1
        new_pos1 = self.encode_pos(new_x1, new_y1)
        if (pos1 == self.source1 and action1 == 1):  # hit the wall
            new_pos1 = pos1

        if (new_pos0 == new_pos1):
            new_pos0 = pos0
            new_pos1 = pos1

        return self.encode_state(new_pos0, new_pos1)

    def print(self):
        map = [['.' for i in range(3)] for j in range(3)]
        [pos0, pos1] = self.decode_state(self.s)
        [x0, y0] = self.decode_pos(pos0)
        [x1, y1] = self.decode_pos(pos1)
        map[y0][x0] = '0'
        map[y1][x1] = '1'
        for i in range(2, -1, -1):
            s = ''
            for j in range(3):
                s += map[i][j]
            print(s)
