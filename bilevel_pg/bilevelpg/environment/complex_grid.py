import bilevel_pg.bilevelpg.environment.discrete_game as discrete
import numpy as np
from bilevel_pg.bilevelpg.environment.base_game import BaseGame
# import utils
import bilevel_pg.bilevelpg.environment.utils as utils

class ComplexGridEnv(discrete.DiscreteEnv):
    def __init__(self):
        self.num_state = 625
        self.num_action = 4
        num_state = self.num_state
        num_action = self.num_action ** 2
        self.agent_num = 2
        self.action_num = 4

        self.source0 = self.encode_pos(0, 4)
        self.source1 = self.encode_pos(4, 4)
        self.target0 = self.encode_pos(4, 0)
        self.target1 = self.encode_pos(0, 0)
        payoff = [[-1, -1] for i in range(num_state)]
        trap_pos = []
        trap_pos.append(self.encode_pos(3, 1))
        trap_pos.append(self.encode_pos(3, 3))
        trap_pos.append(self.encode_pos(1, 1))
        trap_pos.append(self.encode_pos(1, 3))
        con_pos = self.encode_pos(2, 2)
        for i in range(25):
            for tp in trap_pos:
                payoff[self.encode_state(i, tp)][1] = -10
                payoff[self.encode_state(tp, i)][0] = -10
            payoff[self.encode_state(self.target0, i)][0] = 10
            payoff[self.encode_state(i, self.target1)][1] = 10
            payoff[self.encode_state(i, con_pos)][1] = 0
            payoff[self.encode_state(con_pos, i)][0] = 0
        self.payoff = payoff

        init_p = np.zeros(num_state)
        init_p[self.encode_state(self.source0, self.source1)] = 1

        P = {state: {action: []
                     for action in range(num_action)} for state in range(num_state)}

        for state in range(num_state):
            for joint_action in range(num_action):
                [action0, action1] = utils.decode_action(joint_action, self.num_action)
                new_state = self.next_state(state, action0, action1)
                reward = self.payoff[new_state]
                [pos0, pos1] = self.decode_state(new_state)
                if pos0 == self.target0 or pos1 == self.target1:
                    done = True
                else:
                    done = False
                P[state][joint_action].append((1.0, new_state, reward, np.array([done] * 2)))

        discrete.DiscreteEnv.__init__(self, num_state, num_action, P, init_p)

    def encode_state(self, encoded_pos0, encoded_pos1):
        return encoded_pos0 * 25 + encoded_pos1

    def decode_state(self, encoded_state):
        return [encoded_state // 25, encoded_state % 25]

    def encode_pos(self, x, y):
        return y * 5 + x

    def decode_pos(self, pos):
        return [pos % 5, pos // 5]

    def out(self, x, y):
        if x < 0 or  x > 4 or y < 0 or y > 4:
            return True
        else:
            return False

    def next_state(self, cur_state, action0, action1):
        dir_x = [0, 0, -1, 1] # up, down, left, right
        dir_y = [1, -1, 0, 0] # up, down, left, right
        [pos0, pos1] = self.decode_state(cur_state)
        [x0, y0] = self.decode_pos(pos0)
        [x1, y1] = self.decode_pos(pos1)
        newx0 = x0 + dir_x[action0]
        newy0 = y0 + dir_y[action0]
        newx1 = x1 + dir_x[action1]
        newy1 = y1 + dir_y[action1]
        if self.out(newx0, newy0):
            newx0 = x0
            newy0 = y0
        if self.out(newx1, newy1):
            newx1 = x1
            newy1 = y1
        new_pos0 = self.encode_pos(newx0, newy0)
        new_pos1 = self.encode_pos(newx1, newy1)
        if new_pos1 == new_pos0:
            new_pos1 = pos1
            new_pos0 = pos0
        return self.encode_state(new_pos0, new_pos1)


def test():
    dir_x = [0, 0, -1, 1]
    dir_y = [1, -1, 0, 0]
    env = ComplexGridEnv()
    env.reset()
    m = np.zeros((5, 5)) - 1
    m[1][1] = -10
    m[1][3] = -10
    m[3][1] = -10
    m[3][3] = -10
    m[2][2] = 0
    posx0 = 0
    posy0 = 4
    posx1 = 4
    posy1 = 4
    for i in range(500000):

        action0 = np.random.randint(4)
        action1 = np.random.randint(4)
        newx0 = posx0 + dir_x[action0]
        newy0 = posy0 + dir_y[action0]
        newx1 = posx1 + dir_x[action1]
        newy1 = posy1 + dir_y[action1]

        if env.out(newx0, newy0):
            newx0 = posx0
            newy0 = posy0


        if env.out(newx1, newy1):
            newx1 = posx1
            newy1 = posy1

        if newx0 == newx1 and newy0 == newy1:
            newx0 = posx0
            newy0 = posy0
            newx1 = posx1
            newy1 = posy1

        reward0 = m[newx0][newy0]
        reward1 = m[newx1][newy1]

        done = False

        if newx0 == 4 and newy0 == 0:
            done = True
            reward0 = 10
        if newx1 == 0 and newy1 == 0:
            done = True
            reward1 = 10

        [next_state, reward, idone, _] = env.step(utils.encode_action(action0, action1, 4))
        [pos0, pos1] = env.decode_state(next_state)
        [x0, y0] = env.decode_pos(pos0)
        [x1, y1] = env.decode_pos(pos1)
        if not(x0 == newx0 and x1 == newx1 and y0 == newy0 and y1 == newy1 and int(reward0)== int(reward[0]) and int(reward1) == int(reward[1]) and done == idone):
            print(i)
            print(str(action0) + " " + str(action1))
            print(str(x0) + str(y0) + str(x1) + str(y1) + str(int(reward[0])) + str(int(reward[1])) + str(idone))
            print(str(newx0) + str(newy0) + str(newx1) + str(newy1) + str(int(reward0)) + str(int(reward1)) + str(done))
            break

        posx0 = newx0
        posy0 = newy0
        posx1 = newx1
        posy1 = newy1
    print("success")

if __name__ == "__main__":
    test()