
import numpy as np

from gym import Env, spaces
from gym.utils import seeding
from bilevel_pg.bilevelpg.spaces import MAEnvSpec, MASpace
import bilevel_pg.bilevelpg.environment.utils as utils

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):

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
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA

        # self.action_space = MASpace(spaces.Discrete(self.nA))
        # self.observation_space = MASpace(spaces.Discrete(self.nS))
        self.action_spaces = MASpace(tuple(spaces.Discrete(4) for _ in range(2)))
        self.observation_spaces = MASpace(tuple(spaces.Discrete(nS) for _ in range(2)))
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return np.array([[self.s], [self.s]])

    def step(self, a):
        # print('shape:')
        # print(a.shape)
        # print(a)
        a = utils.encode_action(a[0][0], a[1][0], 4)
        # print(self.s, a)
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction = a
        return ([[s]] * 2, r, d, {"prob" : p})