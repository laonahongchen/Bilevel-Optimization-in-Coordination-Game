import numpy as np

# from malib.logger import logger, tabular
from bilevel_pg.bilevelpg.logger import logger, tabular
import tensorflow as tf

num_sample = 10

class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class MASampler(Sampler):
    def __init__(self, agent_num, max_path_length=20, min_pool_size=10e4, batch_size=64, global_reward=False, **kwargs):
        # super(MASampler, self).__init__(**kwargs)
        self.agent_num = agent_num
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size
        self._global_reward = global_reward
        self._path_length = 0
        self._path_return = np.zeros(self.agent_num)
        self._last_path_return = np.zeros(self.agent_num)
        self._max_path_return = np.array([-np.inf] * self.agent_num, dtype=np.float32)
        self._n_episodes = 0
        self._total_samples = 0
        # self.episode_rewards = [0]  # sum of rewards for all agents
        # self.agent_rewards = [[0] for _ in range(self.agent_num)] # individual agent reward
        self.step = 0
        self._current_observation_n = None
        self.env = None
        self.agents = None
        self.count = 0

    def set_policy(self, policies):
        for agent, policy in zip(self.agents, policies):
            agent.policy = policy

    def batch_ready(self):
        enough_samples = max(agent.replay_buffer.size for agent in self.agents) >= self._min_pool_size
        return enough_samples

    def random_batch(self, i):
        return self.agents[i].pool.random_batch(self._batch_size)

    def initialize(self, env, agents):
        self._current_observation_n = None
        self.env = env
        self.agents = agents

    def sample(self, explore=False):
        self.step += 1
        # print(self._current_observation_n)
        if self._current_observation_n is None:
            # print('now updating')
            self._current_observation_n = self.env.reset()
            # print(self._current_observation_n)
        action_n = []

        supplied_observation = []

        # print(self._current_observation_n)
        # print(self._current_observation_n.shape)
        mix_observe_0 = tf.one_hot(self._current_observation_n[0], self.env.num_state)
        if explore:
            #mix_observe_0 = tf.one_hot(self._current_observation_n[0], self.env.num_state)
            action_0 = np.random.randint(0, self.env.action_num)
            action_0 = [action_0]
            #action_0 = self.agents[0].act(mix_observe_0)
            #supplied_observation.append(mix_observe_0)
            action_1 = np.random.randint(0, self.env.action_num)
            action_1 = [action_1]
            #action_n.append(action_0)
            mix_observe_1 = np.hstack((tf.one_hot(self._current_observation_n[1], self.env.num_state), tf.one_hot(action_0, self.env.action_num)))
            #action_1 = self.agents[1].act(mix_observe_1)
            #action_n.append(action_1)
        else:
            '''
            for agent, current_observation in zip(self.agents, self._current_observation_n):
                action = agent.act(current_observation.astype(np.float32))
                action_n.append(np.array(action))
            '''
            #mix_observe_0 = tf.one_hot(self._current_observation_n[0], self.env.num_state)
            
            action_0 = self.agents[0].act(mix_observe_0)
            #supplied_observation.append(mix_observe_0)
            #print(action_0)
            #action_n.append(action_0)
            mix_observe_1 = np.hstack((tf.one_hot(self._current_observation_n[1], self.env.num_state), tf.one_hot(action_0, self.env.action_num)))
            #print(tf.one_hot(self._current_observation_n[1], self.env.num_state))
            #print(mix_observe_1)
            
            action_1 = self.agents[1].act(mix_observe_1)
            print(action_1)
            #supplied_observation.append(mix_observe_1)
            #action_n.append(action_1)

            # print('action shape:')
            # print(action_0.shape, action_1.shape, np.array(action_n).shape)
        supplied_observation.append(mix_observe_0)
        supplied_observation.append(mix_observe_1)
        #print(supplied_observation)
        action_n.append(action_0)
        action_n.append(action_1)
        action_n = np.asarray(action_n)

        next_observation_n, reward_n, done_n, info = self.env.step(action_n)
        #print(reward_n)
        
        # print('done:')
        # print(type(done_n[0]))
        if self._global_reward:
            reward_n = np.array([np.sum(reward_n)] * self.agent_num)

        # if action_n[0] == 0:
        #     print('explore up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     print(action_n)
        #     print(reward_n)

        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32)
        self._total_samples += 1
        for i, agent in enumerate(self.agents):
            opponent_action = action_n[[j for j in range(len(action_n)) if j != i]].flatten()
            # print("supplied observation:")
            # print(supplied_observation)
            # print('agent action:')
            # print(i, action_n[i].shape)
            agent.replay_buffer.add_sample(
                # observation=self._current_observation_n[i].astype(np.float32),
                observation=supplied_observation[i],
                # action=action_n[i].astype(np.float32),
                action = tf.one_hot(action_n[i], self.env.action_num),
                # reward=reward_n[i].astype(np.float32),
                reward = np.float32(reward_n[i]),
                # terminal=done_n[i].astype(np.float32),
                terminal=np.float32(done_n[i]),
                # next_observation=next_observation_n[i].astype(np.float32),
                next_observation=np.float32(next_observation_n[i]),
                # opponent_action=opponent_action.astype(np.float32)
                opponent_action=tf.one_hot(opponent_action, self.env.action_num)
            )

        self._current_observation_n = next_observation_n
        # for i, rew in enumerate(reward_n):
        #     self.episode_rewards[-1] += rew
        #     self.agent_rewards[i][-1] += rew

        if self.step % (25 * 1000) == 0:
            print("steps: {}, episodes: {}, mean episode reward: {}".format(
                        self.step, len(reward_n), np.mean(reward_n[-1000:])))

        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()
            self._max_path_return = np.maximum(self._max_path_return, self._path_return)
            self._mean_path_return = self._path_return / self._path_length
            self._last_path_return = self._path_return
            self._path_length = 0

            self._path_return = np.zeros(self.agent_num)
            self._n_episodes += 1
            self.log_diagnostics()
            logger.log(tabular)
            logger.dump_all()
        else:
            self._current_observation_n = next_observation_n

    def log_diagnostics(self):
        for i in range(self.agent_num):
            tabular.record('max-path-return_agent_{}'.format(i), self._max_path_return[i])
            tabular.record('mean-path-return_agent_{}'.format(i), self._mean_path_return[i])
            tabular.record('last-path-return_agent_{}'.format(i), self._last_path_return[i])
        tabular.record('episodes', self._n_episodes)
        tabular.record('episode_reward', self._n_episodes)
        tabular.record('total-samples', self._total_samples)