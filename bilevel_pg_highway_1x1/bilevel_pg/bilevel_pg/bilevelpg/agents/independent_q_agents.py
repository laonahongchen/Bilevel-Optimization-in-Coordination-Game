# Created by yingwen at 2019-03-15
import tensorflow as tf
import numpy as np
# from malib.agents.base_agent import OffPolicyAgent
# from malib.core import Serializable
# from malib.utils import tf_utils
from bilevel_pg.bilevelpg.agents.base_agents import OffPolicyAgent
from bilevel_pg.bilevelpg.core import Serializable
from bilevel_pg.bilevelpg.utils import tf_utils


class IQAgent(Serializable):
    def __init__(self,
                 env_specs,
                 policy,
                 qf,
                 replay_buffer,
                 policy_optimizer=tf.optimizers.Adam(0.001),
                 qf_optimizer=tf.optimizers.Adam(0.001),
                 exploration_strategy=None,
                 exploration_interval=10,
                 target_update_tau=0.01,
                 target_update_period=10,
                 td_errors_loss_fn=None,
                 gamma=1,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 name='Bilevel_leader',
                 agent_id=-1
                 ):
        self._Serializable__initialize(locals())
        self._agent_id = agent_id
        self._env_specs = env_specs
        self.action_num = 5
        if self._agent_id >= 0:
            observation_space = self._env_specs.observation_space[self._agent_id]
            action_space = self._env_specs.action_space[self._agent_id]
        else:
            observation_space = self._env_specs.observation_space
            action_space = self._env_specs.action_space

        self._exploration_strategy = exploration_strategy

        self._qf_optimizer = qf_optimizer

        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._td_errors_loss_fn = (
                td_errors_loss_fn or tf.losses.Huber)
        self._gamma = gamma
        self._reward_scale = reward_scale
        self._gradient_clipping = gradient_clipping
        self._train_step = 0
        self._exploration_interval = exploration_interval
        self._exploration_status = False

        self.required_experiences = ['observation', 'actions', 'rewards', 'next_observations',
                                     'opponent_actions', 'target_actions']

        self._observation_space = observation_space
        self._action_space = action_space
        self._policy = policy
        self._qf = qf
        self._replay_buffer = replay_buffer
        self._name = name

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def get_policy_np(self,
                         input_tensor):
        return self._policy.get_action_np(input_tensor)

    def act(self, observation):
        mxv = np.zeros(observation.shape[0])
        mxp = np.zeros(observation.shape[0])
        for action_0 in range(self.action_num):
            tot_action_0 = np.array([action_0 for i in range(observation.shape[0])])
            tot_action_0 = tf.one_hot(tot_action_0, self.action_num)
            # actions = opponent_agent.act(np.hstack((observation, tot_action_0)))
            # actions = tf.one_hot(actions, self._action_space.n)
            # print((observation).shape)
            values = self.get_critic_value(np.hstack((observation, tot_action_0)))
            for i in range(observation.shape[0]):
                if values[i] > mxv[i]:
                    mxv[i] = values[i]
                    mxp[i] = action_0
        return mxp.astype(np.int64)

    def init_opt(self):
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def train(self, batch, agent_id, weights=None):
        # if self.train_sequence_length is not None:
        #     if batch['observations'].shape[1] != self.train_sequence_length:
        #         raise ValueError('Invalid sequence length in batch.')
        loss_info = self._train(batch=batch, agent_id=agent_id, weights=weights)
        return loss_info

    def _train(self, batch, agent_id, weights=None):
        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(agent_id,
                                           batch['observations'],
                                           batch['actions'],
                                           batch['target_actions'],
                                           batch['rewards'],
                                           batch['next_observations'],
                                           batch['terminals'],
                                           weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')

        critic_grads = tape.gradient(critic_loss, critic_variables)

        tf_utils.apply_gradients(critic_grads, critic_variables, self._qf_optimizer, self._gradient_clipping)

        self._train_step += 1


        losses = {
            'critic_loss': critic_loss.numpy(),
        }

        return losses

    def get_critic_value(self,
                         input_tensor):
        return self._qf.get_values(input_tensor)

    def critic_loss(self,
                    agent_id,
                    observations,
                    actions,
                    target_actions,
                    rewards,
                    next_observations,
                    terminals,
                    weights=None):
        """Computes the critic loss for DDPG training.
        Args:
          observations: A batch of observations.
          actions: A batch of actions.
          rewards: A batch of rewards.
          next_observations: A batch of next observations.
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
        Returns:
          critic_loss: A scalar critic loss.
        """
        #print(target_actions.shape)
        #print(tf.one_hot(target_actions[:, agent_id], self.action_space.n))
        target_critic_input = np.hstack((next_observations,
                                         tf.one_hot(target_actions[:, agent_id], self.action_num)))

        target_q_values = self._qf.get_values(target_critic_input)

        rewards = rewards.reshape(-1, 1)
        td_targets = tf.stop_gradient(
                self._reward_scale * rewards + (1 - terminals.reshape(-1, 1)) * self._gamma * target_q_values)

        critic_net_input = np.hstack((observations, actions))

        q_values = self._qf.get_values(critic_net_input)

        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)

        # print(critic_loss)

        if weights is not None:
            critic_loss = weights * critic_loss

        critic_loss = tf.reduce_mean(critic_loss)
        # print(critic_loss)
        return critic_loss

