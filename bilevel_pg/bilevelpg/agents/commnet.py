from bilevel_pg.bilevelpg.core import Serializable
import tensorflow as tf
from bilevel_pg.bilevelpg.agents.base_agents import RLAgent

class CommNetAgent(RLAgent, Serializable):
    def __init__(self,
                 env_specs,
                 # policy,
                 # qf,
                 # replay_buffer,
                 policy_optimizer=tf.optimizers.Adam(lr=0.1),
                 qf_optimizer=tf.optimizers.Adam(lr=0.1),
                 exploration_strategy=None,
                 exploration_interval=10,
                 # target_update_tau=0.1,
                 # target_update_period=10,
                 td_errors_loss_fn=None,
                 gamma=0.75,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 name='Bilevel_follower',
                 agent_id=-1
                 ):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space

        # self._exploration_strategy = exploration_strategy

        self._exploration_strategy = None

        # self._target_policy = Serializable.clone(policy, name='target_policy_agent_{}'.format(self._agent_id))
        # self._target_qf = Serializable.clone(qf, name='target_qf_agent_{}'.format(self._agent_id))

        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer

        # self._target_update_tau = target_update_tau
        # self._target_update_period = target_update_period
        self._td_errors_loss_fn = (
                td_errors_loss_fn or tf.losses.Huber)
        # self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'opponent_actions', 'target_actions']

    def act(self, observation):
        pass

    def train(self):
        pass