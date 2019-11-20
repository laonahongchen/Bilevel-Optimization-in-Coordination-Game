# Created by yingwen at 2019-03-15
import tensorflow as tf
import numpy as np
# from malib.agents.base_agent import OffPolicyAgent
# from malib.core import Serializable
# from malib.utils import tf_utils
from bilevel_pg.bilevelpg.agents.base_agents import OffPolicyAgent
from bilevel_pg.bilevelpg.core import Serializable
from bilevel_pg.bilevelpg.utils import tf_utils
from bilevel_pg.bilevelpg.utils.kernel import adaptive_isotropic_gaussian_kernel


class LeaderSoftQAgent(Serializable):
    def __init__(self,
                 env_specs,
                 policy,
                 qf,
                 replay_buffer,
                 policy_optimizer=tf.optimizers.Adam(),
                 qf_optimizer=tf.optimizers.Adam(),
                 exploration_strategy=None,
                 exploration_interval=10,
                 target_update_tau=0.8,
                 target_update_period=10,
                 td_errors_loss_fn=None,
                 gamma=0.75,
                 reward_scale=1.0,
                 gradient_clipping=None,
                 train_sequence_length=None,
                 kernel_fn=adaptive_isotropic_gaussian_kernel,
                 kernel_n_particles=16,
                 kernel_update_ratio=0.5,
                 name='Bilevel_SoftQ_leader',
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
        self._kernel_fn = kernel_fn
        self._kernel_n_particles = kernel_n_particles
        self._kernel_update_ratio = kernel_update_ratio
        self._name = name

        self._EPS = 1e-6

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def replay_buffer(self):
        return self._replay_buffer

    def critic_loss(self,
                    observations,
                    actions,
                    opponent_actions,
                    target_actions,
                    rewards,
                    next_observations,
                    weights=None):
        """Create a minimization operation for Q-function update."""

        target_critic_input = np.hstack((next_observations,target_actions))

        target_q_values = self._qf.get_values(target_critic_input)

        critic_net_input = np.hstack((observations, actions, opponent_actions))

        q_values = self._qf.get_values(critic_net_input)

        # Equation 10:
        next_value = tf.reduce_logsumexp(target_q_values, axis=1)

        # Importance weights add just a constant to the value.
        next_value -= tf.math.log(tf.cast(self._value_n_particles, tf.float32))
        next_value += self._action_space * np.log(2)

        # \hat Q in Equation 11:
        ys = tf.stop_gradient(self._reward_scale * rewards + (
                1 - self._terminals_pl) * self._gamma * next_value)

        # Equation 11:
        bellman_residual = 0.5 * tf.reduce_mean((ys - q_values) ** 2)

        return bellman_residual

    def actor_loss(self, observations, opponent_actions, weights=None):
        """Create a minimization operation for policy update (SVGD)."""

        # actions = self._policy.actions_for(
        #     observations=self._observations_ph,
        #     n_action_samples=self._kernel_n_particles,
        #     reuse=True)

        actions = self._policy.get_actions_np(observations)

        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        n_updated_actions = int(
            self._kernel_n_particles * self._kernel_update_ratio)
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions, updated_actions = tf.split(
            actions, [n_fixed_actions, n_updated_actions], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)

        target_q_input = np.hstack((observations, fixed_actions))

        # svgd_target_values = self._qf.output_for(
        #     self._observations_ph[:, None, :], fixed_actions, reuse=True)

        svgd_target_values = self._qf.get_values(target_q_input)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = tf.reduce_sum(
            tf.math.log(1 - fixed_actions ** 2 + self._EPS), axis=-1)
        log_p = svgd_target_values + squash_correction

        grad_log_p = tf.gradients(log_p, fixed_actions)[0]
        grad_log_p = tf.expand_dims(grad_log_p, axis=2)
        grad_log_p = tf.stop_gradient(grad_log_p)

        kernel_dict = self._kernel_fn(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = tf.expand_dims(kernel_dict["output"], dim=3)

        # Stein Variational Gradient in Equation 13:
        action_gradients = tf.reduce_mean(
            kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)

        # Propagate the gradient through the policy network (Equation 14).
        gradients = tf.gradients(
            updated_actions,
            self._policy.get_params_internal(),
            grad_ys=action_gradients)

        surrogate_loss = tf.reduce_sum([
            tf.reduce_sum(w * tf.stop_gradient(g))
            for w, g in zip(self._policy.get_params_internal(), gradients)
        ])

        return -surrogate_loss

    def get_policy_np(self,
                         input_tensor):
        return self._policy.get_action_np(input_tensor)

    def act(self, observation, opponent_agent):
        mxv = np.zeros(observation.shape[0])
        mxp = np.zeros(observation.shape[0])
        for action_0 in range(self._action_space.n):
            tot_action_0 = np.array([action_0 for i in range(observation.shape[0])])
            tot_action_0 = tf.one_hot(tot_action_0, self._action_space.n)
            actions = opponent_agent.act(np.hstack((observation, tot_action_0)))
            actions = tf.one_hot(actions, self._action_space.n)
            values = self.get_critic_value(np.hstack((observation, tot_action_0, actions)))
            for i in range(observation.shape[0]):
                if values[i] > mxv[i]:
                    mxv[i] = values[i]
                    mxp[i] = action_0
        return mxp.astype(np.int64)

    def init_opt(self):
        self._exploration_status = True

    def init_eval(self):
        self._exploration_status = False

    def train(self, batch, weights=None):
        # if self.train_sequence_length is not None:
        #     if batch['observations'].shape[1] != self.train_sequence_length:
        #         raise ValueError('Invalid sequence length in batch.')
        loss_info = self._train(batch=batch, weights=weights)
        return loss_info

    def _train(self, batch, weights=None):
        critic_variables = self._qf.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_variables, 'No qf variables to optimize.'
            tape.watch(critic_variables)
            critic_loss = self.critic_loss(batch['observations'],
                                           batch['actions'],
                                           batch['opponent_actions'],
                                           batch['target_actions'],
                                           batch['rewards'],
                                           batch['next_observations'],
                                           weights=weights)
        tf.debugging.check_numerics(critic_loss, 'qf loss is inf or nan.')

        critic_grads = tape.gradient(critic_loss, critic_variables)

        # print(critic_grads)

        tf_utils.apply_gradients(critic_grads, critic_variables, self._qf_optimizer, self._gradient_clipping)

        actor_variables = self._policy.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_variables, 'No actor variables to optimize.'
            tape.watch(actor_variables)
            actor_loss = self.actor_loss(batch['observations'], batch['opponent_actions'], weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_variables)
        tf_utils.apply_gradients(actor_grads, actor_variables, self._policy_optimizer, self._gradient_clipping)
        self._train_step += 1

        if self._train_step % self._target_update_period == 0:
            self._update_target()

        losses = {
            'pg_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
        }

        return losses

    def get_critic_value(self,
                         input_tensor):
        return self._qf.get_values(input_tensor)
'''
    def critic_loss(self,
                    observations,
                    actions,
                    opponent_actions,
                    target_actions,
                    rewards,
                    next_observations,
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

        target_critic_input = np.hstack((next_observations,
                                         tf.one_hot(target_actions[:, 0], self.action_space.n),
                                         tf.one_hot(target_actions[:, 1], self.action_space.n)))

        target_q_values = self._qf.get_values(target_critic_input)

        rewards = rewards.reshape(-1, 1)
        td_targets = tf.stop_gradient(
                self._reward_scale * rewards + self._gamma * target_q_values)

        critic_net_input = np.hstack((observations, actions, opponent_actions))

        q_values = self._qf.get_values(critic_net_input)

        critic_loss = self._td_errors_loss_fn(reduction=tf.losses.Reduction.NONE)(td_targets, q_values)

        # print(critic_loss)

        if weights is not None:
            critic_loss = weights * critic_loss

        critic_loss = tf.reduce_mean(critic_loss)
        # print(critic_loss)
        return critic_loss

'''